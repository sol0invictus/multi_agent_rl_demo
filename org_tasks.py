"""Partitioned data tasks: the same questions as tasks.py, but the data lives at several sites.

The main table (orders / employees / sessions) is sharded row-wise across K sites with uneven shares;
the lookup tables (customers / departments / users) are replicated at every site. No single site can
answer the question, and the lead agent that receives the question cannot see any data at all, so
the organisation has to decompose the question, collect partial results and combine them correctly
(e.g. an average needs per-site sums and counts, not per-site averages). Exact duplicate rows always
land on the same site, so "count each only once" can be handled locally.

Test buckets (all disjoint from training seeds):
  seen     training templates, 2-3 sites
  tmpl     held-out question templates (retail, hr), 2-3 sites
  domain   held-out domain (web), 2-3 sites
  wide     training templates, 4 sites (more workers than ever seen in training)
"""
import random
from dataclasses import dataclass, field

import tasks as T

SITE_NAMES = ["atlas", "boreal", "cobalt", "delta", "ember", "fjord"]
TRAIN_SITES = (2, 3)
WIDE_SITES = (4,)


@dataclass
class OrgTask(T.Task):
    sites: dict = field(default_factory=dict)   # site name -> {filename: csv text}
    main: str = ""                                # the sharded table
    shared: list = field(default_factory=list)    # tables replicated at every site

    @property
    def n_sites(self):
        return len(self.sites)


def shard(rng, csv_text, names):
    """Split CSV rows across sites with uneven random shares; identical lines go to the same site."""
    header, *lines = csv_text.rstrip("\n").split("\n")
    w = [rng.uniform(0.5, 2.0) for _ in names]
    where, parts = {}, {n: [] for n in names}
    for ln in lines:
        if ln not in where:
            where[ln] = rng.choices(names, weights=w)[0]
        parts[where[ln]].append(ln)
    if any(not p for p in parts.values()):
        raise T.Retry
    return {n: "\n".join([header] + p) + "\n" for n, p in parts.items()}


def make_org_task(seed, difficulty, templates, site_counts):
    rng = random.Random(seed)
    pool = [t for t in templates if T.TEMPLATES[t][1] == difficulty] or list(templates)
    domains = sorted({T.TEMPLATES[t][0] for t in pool})
    for _ in range(50):
        dom = rng.choice(domains)
        name = rng.choice([t for t in pool if T.TEMPLATES[t][0] == dom])
        x = T.DOMAINS[dom](rng)
        try:
            q, a, kind = T.TEMPLATES[name][2](rng, x)
            files, mess, note = T._messy_files(rng, x, T.TEMPLATES[name][1], exclude=("split",))
            k = rng.choice(site_counts)
            names = sorted(rng.sample(SITE_NAMES, k))
            parts = shard(rng, files[x.main], names)
        except T.Retry:
            continue
        shared = sorted(f for f in files if f != x.main)
        sites = {n: {x.main: parts[n], **{f: files[f] for f in shared}} for n in names}
        if note:
            q = f"{q}\n\nNote: {note}"
        return OrgTask(task_id=f"o{seed}", question=q, answer=str(a), kind=kind,
                       difficulty=T.TEMPLATES[name][1], files=files, template=name, mess=mess,
                       sites=sites, main=x.main, shared=shared)
    raise RuntimeError(f"could not draw a well-posed task for seed {seed}")


def make_org_split(n, seed0, templates=T.TRAIN_TEMPLATES, bucket="train", site_counts=TRAIN_SITES,
                   mix=(("easy", 0.2), ("medium", 0.4), ("hard", 0.4))):
    rng = random.Random(seed0)
    out = []
    for i in range(n):
        r, acc = rng.random(), 0.0
        for diff, p in mix:
            acc += p
            if r < acc:
                break
        t = make_org_task(seed0 * 100000 + i, diff, templates, site_counts)
        t.bucket = bucket
        out.append(t)
    return out


TEST_BUCKETS = {
    "seen": (T.TRAIN_TEMPLATES, TRAIN_SITES),
    "tmpl": (sorted(T.HELDOUT_TEMPLATES), TRAIN_SITES),
    "domain": (T.TEST_BUCKETS["domain"], TRAIN_SITES),
    "wide": (T.TRAIN_TEMPLATES, WIDE_SITES),
}


def make_org_test(n_per_bucket, seed0=199, buckets=tuple(TEST_BUCKETS)):
    return [t for j, b in enumerate(TEST_BUCKETS) if b in buckets
            for t in make_org_split(n_per_bucket, seed0 + j, TEST_BUCKETS[b][0], bucket=b,
                                    site_counts=TEST_BUCKETS[b][1])]


if __name__ == "__main__":
    import io
    from collections import Counter

    import pandas as pd
    tr = make_org_split(500, 21)
    te = make_org_test(40)
    print("train", Counter(t.difficulty for t in tr), Counter(t.n_sites for t in tr),
          "messy", sum(bool(t.mess) for t in tr))
    print("test", Counter((t.bucket, t.n_sites) for t in te))
    assert not {t.template for t in tr} & {t.template for t in te if t.bucket in ("tmpl", "domain")}
    for t in tr[:200] + te:   # shards reassemble into exactly the original table
        whole = pd.read_csv(io.StringIO(t.files[t.main]), dtype=str)
        parts = pd.concat([pd.read_csv(io.StringIO(s[t.main]), dtype=str) for s in t.sites.values()])
        assert len(whole) == len(parts) and Counter(map(tuple, whole.values)) == Counter(map(tuple, parts.values))
    for t in te[::23]:
        sizes = {n: s[t.main].count("\n") - 1 for n, s in t.sites.items()}
        print(f"[{t.bucket}/{t.template}/{t.mess}] sites={sizes} {t.question[:110]!r} -> {t.answer}")
