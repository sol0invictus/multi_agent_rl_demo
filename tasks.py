"""Synthetic data-analysis tasks: a tiny stand-in for SmolDataEnvs.

Each task gets its own randomly generated CSV files, so answers cannot be memorised and
the agent has to inspect and compute. Column names are randomised between a few variants,
so the agent also has to look at the files before writing code.

Questions come from named templates ("<domain>/<name>"). Some templates and one whole domain
(web) are never used for training, so the test set measures three things separately:
  seen     training templates on fresh data
  tmpl     held-out question templates in the training domains (retail, hr)
  domain   a held-out domain with its own files and questions (web)
Medium/hard tasks may also get messy files (duplicate rows, "$1,234.50" money strings,
inconsistently capitalised labels, mixed date formats, the main table split across part files).
Gold answers are always computed on the clean data.
"""
import json
import math
import random
import re
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace

import pandas as pd

REGIONS = ["North", "South", "East", "West", "Central"]
PRODUCTS = ["Widget", "Gadget", "Sprocket", "Gizmo", "Doohickey", "Thingamajig", "Whatsit", "Contraption"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
COUNTRIES = ["USA", "Canada", "Mexico", "Brazil", "Germany", "France", "Japan", "India"]
DEPTS = ["Engineering", "Sales", "Marketing", "Finance", "Support", "Operations", "Legal"]
CITIES = ["Austin", "Boston", "Denver", "Seattle", "Chicago"]
FIRST = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lena", "Max", "Nia",
         "Omar", "Pia", "Raj", "Sara", "Tom", "Uma", "Vic", "Wen", "Yara", "Zed"]
LAST = ["Smith", "Garcia", "Kim", "Patel", "Novak", "Rossi", "Silva", "Ito", "Khan", "Moreau", "Berg", "Diaz"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
PLANS = ["Free", "Basic", "Pro", "Enterprise"]
PAGES = ["home", "search", "product", "cart", "checkout", "account", "help"]
DEVICES = ["mobile", "desktop", "tablet"]


@dataclass
class Task:
    task_id: str
    question: str
    answer: str          # gold answer, as a string
    kind: str            # "number" or "text"
    difficulty: str
    files: dict = field(default_factory=dict)  # filename -> csv text
    template: str = ""
    bucket: str = "train"
    mess: list = field(default_factory=list)

    def to_json(self):
        return asdict(self)


class Retry(Exception):
    """The sampled data makes this question ill-posed (tie, empty subset): draw again."""


def _col(rng, *variants):
    return rng.choice(variants)


def _num(x, nd=2):
    x = float(x)
    if math.isnan(x):
        raise Retry
    return str(round(x, nd))


def _top(s, k=0):
    """Index of the k-th largest value; refuses near-ties so the gold answer is unambiguous."""
    s = s.sort_values(ascending=False)
    if len(s) <= k:
        raise Retry
    v = s.iloc[k]
    for j in (k - 1, k + 1):
        if 0 <= j < len(s) and math.isclose(s.iloc[j], v, rel_tol=1e-4, abs_tol=1e-9):
            raise Retry
    return s.index[k]


# ------------------------------------------------------------------------------------- data
def make_retail(rng):
    c = dict(
        order_id=_col(rng, "order_id", "OrderID", "id"),
        date=_col(rng, "order_date", "date", "OrderDate"),
        region=_col(rng, "region", "Region", "sales_region"),
        product=_col(rng, "product", "product_name", "Product"),
        cust=_col(rng, "customer_id", "cust_id", "CustomerID"),
        qty=_col(rng, "quantity", "qty", "units"),
        price=_col(rng, "unit_price", "price", "UnitPrice"),
        segment=_col(rng, "segment", "customer_segment", "Segment"),
        country=_col(rng, "country", "Country"),
        signup=_col(rng, "signup_year", "joined", "year_joined"),
    )
    n_cust = rng.randint(40, 90)
    custs = pd.DataFrame({
        c["cust"]: [f"C{1000 + i}" for i in range(n_cust)],
        "name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n_cust)],
        c["segment"]: [rng.choice(SEGMENTS) for _ in range(n_cust)],
        c["country"]: [rng.choice(COUNTRIES) for _ in range(n_cust)],
        c["signup"]: [rng.randint(2015, 2024) for _ in range(n_cust)],
    })
    n = rng.randint(250, 500)
    base_price = {p: round(rng.uniform(3, 120), 2) for p in PRODUCTS}
    rows = []
    for i in range(n):
        p = rng.choice(PRODUCTS)
        rows.append({
            c["order_id"]: 50000 + i,
            c["date"]: f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            c["region"]: rng.choice(REGIONS),
            c["product"]: p,
            c["cust"]: rng.choice(custs[c["cust"]].tolist()),
            c["qty"]: rng.randint(1, 20),
            c["price"]: round(base_price[p] * rng.uniform(0.8, 1.2), 2),
        })
    o = pd.DataFrame(rows)
    m = o.merge(custs, on=c["cust"])
    return SimpleNamespace(
        c=c, o=o, cu=custs, rev=o[c["qty"]] * o[c["price"]], month=o[c["date"]].str.slice(5, 7).astype(int),
        m=m, mrev=m[c["qty"]] * m[c["price"]],
        tables={"orders.csv": o, "customers.csv": custs},
        main="orders.csv", money=c["price"], label=("orders.csv", c["region"]), date=c["date"])


def make_hr(rng):
    c = dict(
        emp=_col(rng, "employee_id", "emp_id", "EmployeeID"),
        dept_id=_col(rng, "dept_id", "department_id", "DeptID"),
        salary=_col(rng, "salary", "annual_salary", "Salary"),
        hired=_col(rng, "hire_year", "year_hired", "HireYear"),
        rating=_col(rng, "rating", "perf_score", "performance"),
        dname=_col(rng, "dept_name", "department", "name"),
        city=_col(rng, "city", "location", "office"),
    )
    depts = pd.DataFrame({
        c["dept_id"]: [f"D{10 + i}" for i in range(len(DEPTS))],
        c["dname"]: DEPTS,
        c["city"]: [rng.choice(CITIES) for _ in DEPTS],
    })
    n = rng.randint(150, 350)
    dept_base = {d: rng.uniform(50000, 120000) for d in depts[c["dept_id"]]}
    rows = []
    for i in range(n):
        d = rng.choice(depts[c["dept_id"]].tolist())
        rows.append({
            c["emp"]: 7000 + i,
            "full_name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            c["dept_id"]: d,
            c["salary"]: int(dept_base[d] * rng.uniform(0.6, 1.5) // 100 * 100),
            c["hired"]: rng.randint(2008, 2024),
            c["rating"]: round(rng.uniform(1, 5), 1),
        })
    e = pd.DataFrame(rows)
    return SimpleNamespace(c=c, e=e, d=depts, m=e.merge(depts, on=c["dept_id"]),
                           tables={"employees.csv": e, "departments.csv": depts},
                           main="employees.csv", money=c["salary"], label=("departments.csv", c["dname"]), date=None)


def make_web(rng):
    c = dict(
        sid=_col(rng, "session_id", "SessionID", "id"),
        user=_col(rng, "user_id", "uid", "UserID"),
        ts=_col(rng, "timestamp", "started_at", "ts"),
        page=_col(rng, "page", "landing_page", "Page"),
        dur=_col(rng, "duration_s", "seconds", "session_seconds"),
        device=_col(rng, "device", "platform", "device_type"),
        plan=_col(rng, "plan", "subscription", "tier"),
        country=_col(rng, "country", "Country"),
    )
    n_users = rng.randint(60, 120)
    users = pd.DataFrame({
        c["user"]: [f"U{300 + i}" for i in range(n_users)],
        c["plan"]: [rng.choice(PLANS) for _ in range(n_users)],
        c["country"]: [rng.choice(COUNTRIES) for _ in range(n_users)],
    })
    activity = [rng.random() ** 2 for _ in range(n_users)]   # a few heavy users, many light ones
    plan_dur = {p: rng.uniform(60, 400) for p in PLANS}
    n = rng.randint(300, 600)
    rows = []
    for i in range(n):
        u = rng.choices(range(n_users), weights=activity)[0]
        rows.append({
            c["sid"]: 900000 + i,
            c["user"]: users[c["user"]][u],
            c["ts"]: f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} "
                     f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00",
            c["page"]: rng.choice(PAGES),
            c["dur"]: int(plan_dur[users[c["plan"]][u]] * rng.uniform(0.2, 2.0)),
            c["device"]: rng.choices(DEVICES, weights=[5, 4, 1])[0],
        })
    s = pd.DataFrame(rows)
    return SimpleNamespace(c=c, s=s, u=users, m=s.merge(users, on=c["user"]),
                           hour=s[c["ts"]].str.slice(11, 13).astype(int),
                           tables={"sessions.csv": s, "users.csv": users},
                           main="sessions.csv", money=c["dur"], label=("sessions.csv", c["device"]), date=c["ts"])


DOMAINS = {"retail": make_retail, "hr": make_hr, "web": make_web}

# ---------------------------------------------------------------------------------- templates
TEMPLATES = {}   # "domain/name" -> (domain, difficulty, fn(rng, x) -> (question, answer, kind))


def template(domain, difficulty):
    def deco(fn):
        TEMPLATES[f"{domain}/{fn.__name__}"] = (domain, difficulty, fn)
        return fn
    return deco


R2 = " Round to 2 decimal places."
REV = "revenue (quantity times unit price)"

# retail ----------------------------------------------------------------------------------
@template("retail", "easy")
def count_region(rng, x):
    R = rng.choice(REGIONS)
    return f"How many orders were placed in the {R} region?", str(int((x.o[x.c['region']] == R).sum())), "number"


@template("retail", "easy")
def max_price(rng, x):
    P = rng.choice(PRODUCTS)
    return f"What is the highest unit price ever charged for a {P}?", \
        _num(x.o[x.o[x.c['product']] == P][x.c['price']].max()), "number"


@template("retail", "easy")
def distinct_cust(rng, x):
    return "How many distinct customers placed at least one order?", str(x.o[x.c["cust"]].nunique()), "number"


@template("retail", "easy")
def total_qty_product(rng, x):
    P = rng.choice(PRODUCTS)
    return f"What is the total quantity of {P} units sold across all orders?", \
        str(int(x.o[x.o[x.c['product']] == P][x.c['qty']].sum())), "number"


@template("retail", "easy")
def orders_in_month(rng, x):
    k = rng.randint(1, 12)
    return f"How many orders were placed in {MONTHS[k - 1]} 2024?", str(int((x.month == k).sum())), "number"


@template("retail", "medium")
def rev_region(rng, x):
    R = rng.choice(REGIONS)
    return f"What is the total {REV} from orders in the {R} region?" + R2, \
        _num(x.rev[x.o[x.c['region']] == R].sum()), "number"


@template("retail", "medium")
def top_product_qty(rng, x):
    return "Which product has the highest total quantity sold?", \
        _top(x.o.groupby(x.c["product"])[x.c["qty"]].sum()), "text"


@template("retail", "medium")
def busiest_month(rng, x):
    return "In which month of 2024 were the most orders placed? Answer with the month number (1-12).", \
        str(int(_top(x.month.value_counts()))), "number"


@template("retail", "medium")
def best_region_aov(rng, x):
    return "Which region has the highest average order revenue (quantity times unit price per order)?", \
        _top(x.rev.groupby(x.o[x.c["region"]]).mean()), "text"


@template("retail", "medium")
def avg_qty_product(rng, x):
    P = rng.choice(PRODUCTS)
    return f"What is the average quantity per order for {P}?" + R2, \
        _num(x.o[x.o[x.c['product']] == P][x.c['qty']].mean()), "number"


@template("retail", "medium")
def rev_product_quarter(rng, x):
    P, q = rng.choice(PRODUCTS), rng.randint(1, 4)
    sel = (x.o[x.c["product"]] == P) & ((x.month - 1) // 3 == q - 1)
    return (f"What was the total {REV} from {P} orders in Q{q} 2024 "
            f"({MONTHS[3 * q - 3]} through {MONTHS[3 * q - 1]})?" + R2), _num(x.rev[sel].sum()), "number"


@template("retail", "hard")
def rev_segment(rng, x):
    S = rng.choice(SEGMENTS)
    return f"What is the total {REV} from customers in the '{S}' segment?" + R2, \
        _num(x.mrev[x.m[x.c['segment']] == S].sum()), "number"


@template("retail", "hard")
def segment_avg_qty(rng, x):
    return "Which customer segment has the highest average quantity per order?", \
        _top(x.m.groupby(x.c["segment"])[x.c["qty"]].mean()), "text"


@template("retail", "hard")
def pct_region_product(rng, x):
    R, P = rng.choice(REGIONS), rng.choice(PRODUCTS)
    sub = x.o[x.o[x.c["product"]] == P]
    return f"What percentage of {P} orders were placed in the {R} region? Round to 1 decimal place.", \
        _num(100 * (sub[x.c['region']] == R).mean(), 1), "number"


@template("retail", "hard")
def cust_more_than_k(rng, x):
    k = rng.randint(4, 7)
    return f"How many customers placed more than {k} orders?", \
        str(int((x.o[x.c['cust']].value_counts() > k).sum())), "number"


@template("retail", "hard")
def country_top_rev(rng, x):
    return f"Customers from which country generated the most total {REV}?", \
        _top(x.mrev.groupby(x.m[x.c["country"]]).sum()), "text"


@template("retail", "hard")
def second_product_rev(rng, x):
    return f"Which product generated the second-highest total {REV}?", \
        _top(x.rev.groupby(x.o[x.c["product"]]).sum(), k=1), "text"


@template("retail", "hard")
def top_customer_rev(rng, x):
    return f"Which customer generated the most total {REV}? Answer with their customer ID.", \
        _top(x.rev.groupby(x.o[x.c["cust"]]).sum()), "text"


@template("retail", "hard")
def signup_before_rev(rng, x):
    Y, R = rng.randint(2017, 2022), rng.choice(REGIONS)
    sel = (x.m[x.c["signup"]] < Y) & (x.m[x.c["region"]] == R)
    return (f"What is the total {REV} of orders in the {R} region placed by customers who signed up "
            f"before {Y}?" + R2), _num(x.mrev[sel].sum()), "number"


@template("retail", "hard")
def mom_increase(rng, x):
    monthly = x.rev.groupby(x.month).sum().reindex(range(1, 13), fill_value=0.0)
    return ("Which month of 2024 saw the largest increase in total revenue (quantity times unit price) "
            "compared with the previous month? Answer with the month number (2-12)."), \
        str(int(_top(monthly.diff().dropna()))), "number"


@template("retail", "hard")
def avg_distinct_products(rng, x):
    return ("Among customers who placed at least one order, what is the average number of distinct "
            "products each customer has ordered?" + R2), \
        _num(x.o.groupby(x.c["cust"])[x.c["product"]].nunique().mean()), "number"


# hr ----------------------------------------------------------------------------------------
@template("hr", "easy")
def n_emp(rng, x):
    return "How many employees are listed in total?", str(len(x.e)), "number"


@template("hr", "easy")
def max_salary(rng, x):
    return "What is the highest salary of any employee?", str(int(x.e[x.c['salary']].max())), "number"


@template("hr", "easy")
def n_hired_after(rng, x):
    Y = rng.randint(2012, 2020)
    return f"How many employees were hired after {Y} (not including {Y})?", \
        str(int((x.e[x.c['hired']] > Y).sum())), "number"


@template("hr", "easy")
def min_rating_year(rng, x):
    Y = rng.randint(2010, 2022)
    sub = x.e[x.e[x.c["hired"]] == Y]
    if sub.empty:
        raise Retry
    return f"What is the lowest performance rating of any employee hired in {Y}?", \
        _num(sub[x.c['rating']].min(), 1), "number"


@template("hr", "medium")
def n_dept(rng, x):
    D = rng.choice(DEPTS)
    return f"How many employees work in the {D} department?", str(int((x.m[x.c['dname']] == D).sum())), "number"


@template("hr", "medium")
def avg_rating(rng, x):
    Y = rng.randint(2012, 2020)
    return f"What is the average performance rating of employees hired in or after {Y}?" + R2, \
        _num(x.e[x.e[x.c['hired']] >= Y][x.c['rating']].mean()), "number"


@template("hr", "medium")
def avg_salary_dept(rng, x):
    D = rng.choice(DEPTS)
    return f"What is the average salary in the {D} department?" + R2, \
        _num(x.m[x.m[x.c['dname']] == D][x.c['salary']].mean()), "number"


@template("hr", "medium")
def n_city(rng, x):
    city = rng.choice(sorted(set(x.d[x.c["city"]])))
    return f"How many employees work in departments whose office is in {city}?", \
        str(int((x.m[x.c['city']] == city).sum())), "number"


@template("hr", "hard")
def top_median_dept(rng, x):
    return "Which department (by name) has the highest median salary?", \
        _top(x.m.groupby(x.c["dname"])[x.c["salary"]].median()), "text"


@template("hr", "hard")
def dept_avg_after(rng, x):
    D, Y = rng.choice(DEPTS), rng.randint(2012, 2020)
    sub = x.m[(x.m[x.c["dname"]] == D) & (x.m[x.c["hired"]] > Y)]
    if sub.empty:
        raise Retry
    return f"What is the average salary of {D} employees hired after {Y}?" + R2, \
        _num(sub[x.c['salary']].mean()), "number"


@template("hr", "hard")
def top_city_payroll(rng, x):
    return "Which office city has the largest total salary payroll?", \
        _top(x.m.groupby(x.c["city"])[x.c["salary"]].sum()), "text"


@template("hr", "hard")
def rating_gap(rng, x):
    a = x.m.groupby(x.c["dname"])[x.c["rating"]].mean()
    return ("What is the difference between the highest and the lowest department-average performance "
            "rating?" + R2), _num(a.max() - a.min()), "number"


@template("hr", "hard")
def top_hire_year_dept(rng, x):
    D = rng.choice(DEPTS)
    return f"In which year were the most {D} employees hired?", \
        str(int(_top(x.m[x.m[x.c['dname']] == D][x.c['hired']].value_counts()))), "number"


@template("hr", "hard")
def above_dept_avg(rng, x):
    avg = x.e.groupby(x.c["dept_id"])[x.c["salary"]].transform("mean")
    return "How many employees earn strictly more than the average salary of their own department?", \
        str(int((x.e[x.c['salary']] > avg).sum())), "number"


# web (held-out domain) ------------------------------------------------------------------------
@template("web", "easy")
def n_sessions_device(rng, x):
    dv = rng.choice(DEVICES)
    return f"How many sessions were on {dv} devices?", str(int((x.s[x.c['device']] == dv).sum())), "number"


@template("web", "easy")
def max_duration(rng, x):
    pg = rng.choice(PAGES)
    return f"What is the longest session duration, in seconds, among sessions on the '{pg}' page?", \
        str(int(x.s[x.s[x.c['page']] == pg][x.c['dur']].max())), "number"


@template("web", "easy")
def n_users_plan(rng, x):
    pl = rng.choice(PLANS)
    return f"How many users are on the {pl} plan?", str(int((x.u[x.c['plan']] == pl).sum())), "number"


@template("web", "medium")
def avg_duration_device(rng, x):
    dv = rng.choice(DEVICES)
    return f"What is the average session duration in seconds on {dv} devices?" + R2, \
        _num(x.s[x.s[x.c['device']] == dv][x.c['dur']].mean()), "number"


@template("web", "medium")
def top_page(rng, x):
    return "Which page has the most sessions?", _top(x.s[x.c["page"]].value_counts()), "text"


@template("web", "medium")
def busiest_hour(rng, x):
    return "During which hour of the day (0-23) did the most sessions start?", \
        str(int(_top(x.hour.value_counts()))), "number"


@template("web", "medium")
def pct_device(rng, x):
    dv = rng.choice(DEVICES)
    return f"What percentage of all sessions were on {dv} devices? Round to 1 decimal place.", \
        _num(100 * (x.s[x.c['device']] == dv).mean(), 1), "number"


@template("web", "hard")
def pct_users_page(rng, x):
    pg = rng.choice(PAGES)
    seen = x.s[x.s[x.c["page"]] == pg][x.c["user"]].nunique()
    return (f"What percentage of all users in users.csv had at least one session on the '{pg}' page? "
            "Round to 1 decimal place."), _num(100 * seen / len(x.u), 1), "number"


@template("web", "hard")
def plan_sessions_per_user(rng, x):
    per_user = x.s[x.c["user"]].value_counts().reindex(x.u[x.c["user"]], fill_value=0)
    by_plan = per_user.groupby(x.u.set_index(x.c["user"])[x.c["plan"]]).mean()
    return ("Which plan has the highest average number of sessions per user (users with no sessions "
            "count as zero)?"), _top(by_plan), "text"


@template("web", "hard")
def country_total_duration(rng, x):
    return "Users from which country spent the most total time (sum of session durations)?", \
        _top(x.m.groupby(x.c["country"])[x.c["dur"]].sum()), "text"


@template("web", "hard")
def users_more_than_k_device(rng, x):
    dv, k = rng.choice(["mobile", "desktop"]), rng.randint(2, 5)
    sub = x.s[x.s[x.c["device"]] == dv]
    return f"How many users had more than {k} sessions on {dv} devices?", \
        str(int((sub[x.c['user']].value_counts() > k).sum())), "number"


# ------------------------------------------------------------------------------------ splits
HELDOUT_DOMAINS = {"web"}
HELDOUT_TEMPLATES = {"retail/orders_in_month", "retail/rev_product_quarter", "retail/mom_increase",
                     "retail/avg_distinct_products", "hr/min_rating_year", "hr/n_city", "hr/above_dept_avg"}
TRAIN_TEMPLATES = sorted(t for t, (d, _, _) in TEMPLATES.items()
                         if d not in HELDOUT_DOMAINS and t not in HELDOUT_TEMPLATES)
TEST_BUCKETS = {
    "seen": TRAIN_TEMPLATES,
    "tmpl": sorted(HELDOUT_TEMPLATES),
    "domain": sorted(t for t, (d, _, _) in TEMPLATES.items() if d in HELDOUT_DOMAINS),
}
MESS_P = {"easy": 0.0, "medium": 0.5, "hard": 0.9}
MESS_KINDS = ["dup", "money", "case", "dates", "split"]
MON3 = [m[:3] for m in MONTHS]


def _fmt_date(rng, v):
    """'2024-03-05[ 14:23:00]' -> one of several equivalent spellings (numeric ones are month/day)."""
    d, _, t = v.partition(" ")
    y, mo, da = d.split("-")
    t = t[:5]
    f = rng.randrange(4)
    if f == 0:
        out = d
    elif f == 1:
        out = f"{mo}/{da}/{y}"
    elif f == 2:
        out = f"{MON3[int(mo) - 1]} {int(da)}, {y}"
    else:
        out = f"{int(da)} {MONTHS[int(mo) - 1]} {y}"
    return f"{out} {t}" if t else out


def _messy_files(rng, x, difficulty, exclude=()):
    """CSV text for each table, possibly dirtied. Returns (files, mess kinds, note for the question)."""
    tables = {k: v.copy() for k, v in x.tables.items()}
    kinds, notes = [], []
    if rng.random() < MESS_P[difficulty]:
        pool = [k for k in MESS_KINDS if (k != "dates" or x.date) and k not in exclude]
        kinds = rng.sample(pool, 2 if difficulty == "hard" and rng.random() < 0.6 else 1)
    main_desc = x.main
    if "split" in kinds:
        main_desc = f"the {x.main[:-4]}_part*.csv files"
    for k in sorted(kinds, key=lambda k: k == "dup"):   # dup after case/dates so copies stay identical
        if k == "dup":
            t = tables[x.main]
            extra = t.sample(n=max(3, len(t) // 20), random_state=rng.randint(0, 2**31))
            tables[x.main] = pd.concat([t, extra]).sample(frac=1, random_state=rng.randint(0, 2**31))
            notes.append(f"Some rows in {main_desc} are exact duplicates of other rows; count each only once.")
        elif k == "money":
            fname = next(f for f, t in tables.items() if x.money in t.columns)
            col = tables[fname][x.money]
            fmt = (lambda v: f"{int(v)}s") if fname == "sessions.csv" else (lambda v: f"${v:,.2f}")
            tables[fname][x.money] = col.map(fmt).astype(object)
        elif k == "case":
            fname, colname = x.label
            vary = [str, str.lower, str.upper, lambda s: " " + s, lambda s: s + " "]
            tables[fname][colname] = tables[fname][colname].map(lambda v: rng.choice(vary)(v))
            where = main_desc if fname == x.main else fname
            notes.append(f"Values in the {colname} column of {where} are inconsistently capitalized "
                         "and may have stray spaces.")
        elif k == "dates":
            tables[x.main][x.date] = tables[x.main][x.date].map(lambda v: _fmt_date(rng, v))
            notes.append(f"The {x.date} column mixes several date formats; numeric dates are month/day/year.")
    if "split" in kinds:   # last, so duplicates can land in different parts
        t = tables.pop(x.main)
        n_parts = rng.randint(2, 4)
        cuts = sorted(rng.sample(range(1, len(t)), n_parts - 1))
        for i, (lo, hi) in enumerate(zip([0] + cuts, cuts + [len(t)])):
            tables[f"{x.main[:-4]}_part{i + 1}.csv"] = t.iloc[lo:hi]
    return {f: t.to_csv(index=False) for f, t in tables.items()}, kinds, " ".join(notes)


def make_task(seed, difficulty, templates):
    rng = random.Random(seed)
    pool = [t for t in templates if TEMPLATES[t][1] == difficulty] or list(templates)
    domains = sorted({TEMPLATES[t][0] for t in pool})
    for _ in range(50):
        dom = rng.choice(domains)
        name = rng.choice([t for t in pool if TEMPLATES[t][0] == dom])
        x = DOMAINS[dom](rng)
        try:
            q, a, kind = TEMPLATES[name][2](rng, x)
        except Retry:
            continue
        files, mess, note = _messy_files(rng, x, TEMPLATES[name][1])
        if note:
            q = f"{q}\n\nNote: {note}"
        return Task(task_id=f"t{seed}", question=q, answer=str(a), kind=kind, difficulty=TEMPLATES[name][1],
                    files=files, template=name, mess=mess)
    raise RuntimeError(f"could not draw a well-posed task for seed {seed}")


def make_split(n, seed0, templates=TRAIN_TEMPLATES, bucket="train",
               mix=(("easy", 0.1), ("medium", 0.3), ("hard", 0.6))):
    rng = random.Random(seed0)
    out = []
    for i in range(n):
        r, acc = rng.random(), 0.0
        for diff, p in mix:
            acc += p
            if r < acc:
                break
        t = make_task(seed0 * 100000 + i, diff, templates)
        t.bucket = bucket
        out.append(t)
    return out


def make_test(n_per_bucket, seed0=99):
    return [t for j, (b, tmpl) in enumerate(TEST_BUCKETS.items())
            for t in make_split(n_per_bucket, seed0 + j, tmpl, bucket=b)]


# ---------------------------------------------------------------- verifier
_NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+")


def grade(answer, task):
    """Return True if `answer` (free text from the agent) matches the gold answer."""
    if answer is None:
        return False
    a = str(answer).strip().strip("`'\"").strip()
    if not a:
        return False
    if task.kind == "number":
        gold = float(task.answer)
        nums = _NUM.findall(a.replace("$", ""))
        if not nums:
            return False
        try:
            val = float(nums[-1].replace(",", ""))   # last number in the answer
        except ValueError:
            return False
        return math.isclose(val, gold, rel_tol=1e-3, abs_tol=0.011)
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    return norm(a) == norm(task.answer) or norm(a).endswith(" " + norm(task.answer)) and len(norm(a)) < 40


if __name__ == "__main__":
    from collections import Counter
    print(len(TEMPLATES), "templates;", len(TRAIN_TEMPLATES), "train;",
          {b: len(t) for b, t in TEST_BUCKETS.items()})
    tr = make_split(2000, 11)
    te = make_test(40)
    print("train difficulty", Counter(t.difficulty for t in tr), "messy", sum(bool(t.mess) for t in tr))
    print("test buckets", Counter(t.bucket for t in te))
    assert not {t.template for t in tr} & {t.template for t in te if t.bucket != "seen"}
    for t in te[::13]:
        print(f"[{t.bucket}/{t.template}/{t.mess}] {t.question} -> {t.answer}")
