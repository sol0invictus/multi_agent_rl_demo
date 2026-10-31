"""Multi-agent harnesses. Every agent is the same policy behind its own OpenAI-compatible URL.

  org    A lead plus one data specialist (worker) per site. The lead receives the question but has no
         data access; its only actions are ask(site, request) and submit(answer). Several asks in one
         turn run in parallel. A worker sees only its own site's files, keeps its conversation across
         requests, works with list_files / read_file / run_python, and answers with reply(message).
         Roles differ only by context (system prompt, tools, history), as in Avalon-M1: the policy
         weights are shared, so behaviour that wins as a worker or as a lead is reinforced in one model.
  solo   The single-agent baseline: one agent, all sites' files in one workspace (site/<file>),
         pytools-style tools. Same information, no communication.

Each harness returns {"answer", "status", "agents": [agent ids], "roles": {agent id: role}, ...stats}.
Latency is counted in model calls along the critical path (parallel asks cost the slowest one),
a stand-in for Avalon's wall-clock.
"""
import asyncio
import json

from openai import AsyncOpenAI, BadRequestError

LEAD_MAX_TURNS = 6
WORKER_MAX_TURNS = 6     # model calls per request
SOLO_MAX_TURNS = 10


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


def _client(base_url):
    return AsyncOpenAI(base_url=base_url, api_key="x", max_retries=0, timeout=600)


async def _chat(client, **kw):
    try:
        return (await client.chat.completions.create(model="policy", **kw)).choices[0]
    except BadRequestError:   # context window exceeded
        return None


def _args(tc):
    try:
        a = json.loads(tc.function.arguments)
        return a if isinstance(a, dict) else {}
    except json.JSONDecodeError:
        return {}


def _layout(task):
    k = task.n_sites
    return (f"{task.main} holds that site's own share of the rows (the {k} shares do not overlap; together they "
            f"form the full table)" + "".join(f", and {f} is a complete copy, identical at every site"
                                              for f in task.shared))


DATA_TOOLS = [
    _fn("list_files", "List the files in the working directory.", {}, []),
    _fn("read_file", "Show the first lines of a text file.",
        {"path": {"type": "string"}, "max_lines": {"type": "integer", "description": "Default 10."}}, ["path"]),
    _fn("run_python", "Execute a Python script and return stdout/stderr. pandas is available.",
        {"code": {"type": "string"}}, ["code"]),
]


async def _data_tool(ws, name, args):
    """Run one data tool in a workspace; None if `name` is not a data tool."""
    if name == "list_files":
        return (await ws.bash("ls -la; ls -d */ 2>/dev/null | xargs -r ls -la"))[1]
    if name == "read_file" and "path" in args:
        k = args.get("max_lines", 10)
        k = k if isinstance(k, int) and 0 < k <= 200 else 10
        return (await ws.argv(["head", "-n", str(k), str(args["path"])]))[1]
    if name == "run_python" and "code" in args:
        with open(f"{ws.host}/_script.py", "w") as f:
            f.write(str(args["code"]))
        _, out = await ws.argv(["python3", "_script.py"])
        return out or "(no output; use print() to see values)"
    return None


# ------------------------------------------------------------------------------------------ org
LEAD_SYS = ("You lead a small analytics organisation. The data is split across several sites, and each site has "
            "a data specialist who can inspect and compute on that site's files. You cannot see any data "
            "yourself. Send specialists requests with ask(site, request); you may ask several sites in the same "
            "turn and they work in parallel. Combine their results, then call submit with the final answer.")
LEAD_TOOLS = [
    _fn("ask", "Send a request to the data specialist of one site and get their reply.",
        {"site": {"type": "string", "description": "Site name."},
         "request": {"type": "string", "description": "What you need from that site."}}, ["site", "request"]),
    _fn("submit", "Submit the final answer to the question and end the task.",
        {"answer": {"type": "string", "description": "The final answer value."}}, ["answer"]),
]
WORKER_SYS = ("You are the data specialist at site {site} of an analytics organisation with {k} sites ({names}). "
              "Your working directory holds this site's files only: {layout}. The other sites hold the rest of "
              "the data. The lead of the organisation sends you requests. Inspect the files and compute with "
              "run_python (print what you need to see), then call reply with your answer for the lead.")
WORKER_TOOLS = DATA_TOOLS + [
    _fn("reply", "Send your answer to the lead and finish this request.",
        {"message": {"type": "string"}}, ["message"]),
]


class Worker:
    def __init__(self, site, task, ws, base_url):
        self.site, self.ws, self.client = site, ws, _client(base_url)
        self.lock = asyncio.Lock()   # two asks to one site in a turn run one after the other
        self.msgs = [{"role": "system", "content": WORKER_SYS.format(
            site=site, k=task.n_sites, names=", ".join(task.sites), layout=_layout(task))}]
        self.calls = self.tool_calls = self.fallbacks = self.exhausted = 0

    async def handle(self, request):
        """Serve one request from the lead. Returns (reply text, model calls used)."""
        async with self.lock:
            self.msgs.append({"role": "user", "content": f"Request from the lead: {request}"})
            for i in range(WORKER_MAX_TURNS):
                ch = await _chat(self.client, messages=self.msgs, tools=WORKER_TOOLS)
                self.calls += 1
                if ch is None:
                    self.exhausted += 1
                    return f"[{self.site} could not answer: its context is full]", i + 1
                m = ch.message
                if not m.tool_calls:   # a plain message is taken as the reply
                    self.fallbacks += 1
                    self.msgs.append({"role": "assistant", "content": m.content or ""})
                    return m.content or "", i + 1
                self.msgs.append({"role": "assistant", "content": m.content or "",
                                  "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
                reply = None
                for tc in m.tool_calls:
                    args, name = _args(tc), tc.function.name
                    if name == "reply" and reply is None:
                        reply = str(args.get("message", ""))
                        res = "Reply delivered to the lead."
                    else:
                        self.tool_calls += 1
                        res = await _data_tool(self.ws, name, args)
                        res = res if res is not None else f"Error: unknown tool or bad arguments: {name}"
                    self.msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res})
                if reply is not None:
                    return reply, i + 1
            self.exhausted += 1
            return f"[{self.site} ran out of steps before replying]", WORKER_MAX_TURNS


def _lead_prompt(task):
    return (f"Sites: {', '.join(task.sites)}.\nAt each site, {_layout(task)}.\n\n"
            f"Question (about the combined data of all sites): {task.question}\n\n"
            "Report just the final value.")


async def org_harness(task, workspaces, url_for):
    """workspaces: site -> Workspace; url_for(agent id) -> base_url of that agent's capture endpoint."""
    lead = _client(url_for("lead"))
    workers = {s: Worker(s, task, workspaces[s], url_for(s)) for s in task.sites}
    msgs = [{"role": "system", "content": LEAD_SYS}, {"role": "user", "content": _lead_prompt(task)}]
    st = {"lead_turns": 0, "asks": 0, "bad_asks": 0, "fanouts": [], "latency": 0}
    asked = set()

    def done(answer, status):
        ws = workers.values()
        return {"answer": answer, "status": status, "agents": ["lead", *workers],
                "roles": {"lead": "lead", **{s: "worker" for s in workers}},
                "n_sites": task.n_sites, "coverage": len(asked) / task.n_sites,
                "max_fanout": max(st["fanouts"], default=0),
                "worker_calls": sum(w.calls for w in ws), "worker_tool_calls": sum(w.tool_calls for w in ws),
                "reply_fallbacks": sum(w.fallbacks for w in ws), "worker_exhausted": sum(w.exhausted for w in ws),
                "n_tool_calls": st["asks"] + sum(w.tool_calls for w in ws),
                **{k: v for k, v in st.items() if k != "fanouts"}}

    for _ in range(LEAD_MAX_TURNS):
        ch = await _chat(lead, messages=msgs, tools=LEAD_TOOLS)
        st["lead_turns"] += 1
        st["latency"] += 1
        if ch is None:
            return done(None, "context")
        m = ch.message
        if not m.tool_calls:
            return done(m.content, "final_text")
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
        for tc in m.tool_calls:
            if tc.function.name == "submit":
                return done(str(_args(tc).get("answer", "")), "submitted")
        jobs, results = [], {}
        for tc in m.tool_calls:
            args = _args(tc)
            site = str(args.get("site", "")).strip().lower()
            if tc.function.name == "ask" and site in workers and args.get("request"):
                jobs.append((tc.id, site, str(args["request"])))
            else:
                st["bad_asks"] += 1
                results[tc.id] = ((f"Error: unknown site {site!r}; sites are {', '.join(workers)}."
                                   if tc.function.name == "ask" else f"Error: unknown tool {tc.function.name}"), 0)
        outs = await asyncio.gather(*(workers[s].handle(r) for _, s, r in jobs))
        per_site = {}
        for (tid, site, _), (reply, n) in zip(jobs, outs):
            results[tid] = (f"[{site}] {reply}", n)
            per_site[site] = per_site.get(site, 0) + n    # asks to one site are serialised
            asked.add(site)
        st["asks"] += len(jobs)
        st["fanouts"].append(len(per_site))
        st["latency"] += max(per_site.values(), default=0)
        for tc in m.tool_calls:
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": results[tc.id][0]})
    return done(None, "max_turns")


# ----------------------------------------------------------------------------------------- solo
SOLO_SYS = ("You are a data analysis assistant. Explore the data with list_files and read_file, compute results "
            "with run_python (print what you need to see), and finish by calling final_answer.")
SOLO_TOOLS = DATA_TOOLS + [
    _fn("final_answer", "Report the final answer and end the task.", {"answer": {"type": "string"}}, ["answer"]),
]


def _solo_prompt(task):
    return (f"The data is split across {task.n_sites} sites, one subdirectory each: {', '.join(task.sites)}. "
            f"In each site directory, {_layout(task)}.\n\nQuestion (about the combined data of all sites): "
            f"{task.question}\n\nInspect the files, compute the answer, and report just the final value.")


async def solo_harness(task, ws, url_for):
    client = _client(url_for("solo"))
    msgs = [{"role": "system", "content": SOLO_SYS}, {"role": "user", "content": _solo_prompt(task)}]
    n = 0

    def done(answer, status, turns):
        return {"answer": answer, "status": status, "agents": ["solo"], "roles": {"solo": "solo"},
                "n_sites": task.n_sites, "n_tool_calls": n, "latency": turns}

    for t in range(SOLO_MAX_TURNS):
        ch = await _chat(client, messages=msgs, tools=SOLO_TOOLS)
        if ch is None:
            return done(None, "context", t + 1)
        m = ch.message
        if not m.tool_calls:
            return done(m.content, "final_text", t + 1)
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
        for tc in m.tool_calls:
            args, name = _args(tc), tc.function.name
            if name == "final_answer":
                return done(str(args.get("answer", "")), "submitted", t + 1)
            n += 1
            res = await _data_tool(ws, name, args)
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": res if res is not None else f"Error: unknown tool or bad arguments: {name}"})
    return done(None, "max_turns", SOLO_MAX_TURNS)
