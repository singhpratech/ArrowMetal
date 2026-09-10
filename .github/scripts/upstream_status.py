"""Refresh the upstream tracker's live state: what each project did with each report we filed.

Reads the report links out of docs/UPSTREAM.md (the `Report` column of the tracker table), asks
GitHub for each issue's state, the project side's comments (accounts with an OWNER, MEMBER,
COLLABORATOR or CONTRIBUTOR association), label changes, close and reopen events, and pull requests
in the project's own repository that reference the issue. Each such pull request (and a report that
is itself a pull request) is followed too: project-side reviews and comments on it count as activity
on the report, one event per review pass. The reporter's own comments are counted but not shown.
The result is one JSON file the website fetches; nothing on the page talks to api.github.com. Feedback Assistant reports have no public state and are not in the file.

Usage: GITHUB_TOKEN=... upstream_status.py [--md docs/UPSTREAM.md] [--out upstream/status.json]
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
PROJECT_SIDE = {"OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"}
PULSE_DAYS = 14
TEXT_LIMIT = 280


def get(url, token):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                                               **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode()), r.headers
    except urllib.error.HTTPError as e:
        print(f"warning: {url}: HTTP {e.code}", file=sys.stderr)
        return None, None


def paged(url, token):
    out, page = [], 1
    while True:
        data, headers = get(f"{url}&page={page}", token)
        if not data:
            break
        out.extend(data)
        if len(data) < 100 or not headers or 'rel="next"' not in (headers.get("Link") or ""):
            break
        page += 1
    return out


def report_rows(md_path):
    """Yield (project, finding, url) for every GitHub link in the tracker table's Report column."""
    in_table = False
    for line in open(md_path, encoding="utf-8"):
        if line.startswith("| Project |"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table or not line.startswith("| ") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) != 5:
            continue
        project, finding, _evidence, report, _status = cells
        for m in re.finditer(r"\((https://github\.com/[^/]+/[^/]+/(?:issues|pull)/\d+)(#[^)]*)?\)", report):
            yield project, finding, m.group(1), m.group(2) or ""
            break   # the first link in the cell is our report; the rest are related issues


def is_ours_or_bot(user, mine):
    login = (user or {}).get("login") or ""
    return not login or login in mine or (user or {}).get("type") == "Bot" or login.endswith("[bot]")


def pr_activity(owner, repo, num, token, mine, with_comments):
    """Project-side reviews (and, unless the caller already read them, comments) on a pull request.
    Review submissions by one person within half an hour are one event: a review pass."""
    pr_url = f"https://github.com/{owner}/{repo}/pull/{num}"
    inline = {}
    for c in paged(f"{API}/repos/{owner}/{repo}/pulls/{num}/comments?per_page=100", token):
        inline.setdefault(c.get("pull_request_review_id"), []).append(c)
    passes = []
    for rv in sorted(paged(f"{API}/repos/{owner}/{repo}/pulls/{num}/reviews?per_page=100", token), key=lambda r: r.get("submitted_at") or ""):
        if is_ours_or_bot(rv.get("user"), mine) or rv.get("author_association", "NONE") not in PROJECT_SIDE:
            continue
        notes = inline.get(rv["id"], [])
        ev = {"type": "review", "at": rv.get("submitted_at"), "actor": rv["user"]["login"], "association": rv["author_association"].lower(),
              "state": (rv.get("state") or "commented").lower(), "text": clean(rv.get("body")) or (clean(notes[0]["body"]) if notes else ""),
              "inline": len(notes), "url": rv.get("html_url") or pr_url, "pr": pr_url}
        prev = passes[-1] if passes else None
        if prev and prev["actor"] == ev["actor"] and ev["at"] and prev["at"] and \
                (dt.datetime.fromisoformat(ev["at"].replace("Z", "+00:00")) - dt.datetime.fromisoformat(prev["at"].replace("Z", "+00:00"))).total_seconds() <= 1800:
            prev["inline"] += ev["inline"]
            prev["text"] = prev["text"] or ev["text"]
            if ev["state"] in ("approved", "changes_requested"):
                prev["state"] = ev["state"]
            continue
        passes.append(ev)
    if with_comments:
        for c in paged(f"{API}/repos/{owner}/{repo}/issues/{num}/comments?per_page=100", token):
            if is_ours_or_bot(c.get("user"), mine) or c.get("author_association", "NONE") not in PROJECT_SIDE:
                continue
            passes.append({"type": "pr_comment", "at": c["created_at"], "actor": c["user"]["login"], "association": c["author_association"].lower(),
                           "text": clean(c["body"]), "url": c["html_url"], "pr": pr_url})
    return passes


def clean(text):
    text = re.sub(r"```.*?```", "[code]", text or "", flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:TEXT_LIMIT] + ("…" if len(text) > TEXT_LIMIT else "")


def build(md_path, token, ours=("singhpratech",)):
    items = []
    now = dt.datetime.now(dt.timezone.utc)
    for project, finding, url, anchor in report_rows(md_path):
        owner, repo, kind, num = re.match(r"https://github\.com/([^/]+)/([^/]+)/(issues|pull)/(\d+)", url).groups()
        issue, _ = get(f"{API}/repos/{owner}/{repo}/issues/{num}", token)
        if issue is None:
            continue
        reporter = issue["user"]["login"]
        by_us = reporter in ours
        mine = set(ours) | {reporter}          # our accounts and the issue's author both count as "the reporter"
        events, own_comments, community = [], 0, 0
        for c in paged(f"{API}/repos/{owner}/{repo}/issues/{num}/comments?per_page=100", token):
            login = c["user"]["login"]
            if login in mine:
                own_comments += 1
                continue
            if c["user"].get("type") == "Bot" or login.endswith("[bot]"):
                continue
            if c.get("author_association", "NONE") in PROJECT_SIDE:
                events.append({"type": "comment", "at": c["created_at"], "actor": login,
                               "association": c["author_association"].lower(), "text": clean(c["body"]), "url": c["html_url"]})
            else:
                community += 1
        for ev in paged(f"{API}/repos/{owner}/{repo}/issues/{num}/timeline?per_page=100", token):
            at, actor, t = ev.get("created_at"), (ev.get("actor") or {}).get("login"), ev.get("event")
            if actor in mine and t != "cross-referenced":
                continue        # our own labels/closes are not project-side activity; our own fix PRs are shown
            if t in ("labeled", "unlabeled"):
                events.append({"type": t, "at": at, "actor": actor, "label": ev["label"]["name"]})
            elif t in ("closed", "reopened"):
                events.append({"type": t, "at": at, "actor": actor, "reason": ev.get("state_reason"),
                               "commit": (ev.get("commit_url") or "").replace(API + "/repos/", "https://github.com/").replace("/commits/", "/commit/") or None})
            elif t == "cross-referenced" and ev.get("source", {}).get("issue", {}).get("pull_request") \
                    and ev["source"]["issue"].get("repository", {}).get("full_name", "").lower() == f"{owner}/{repo}".lower():
                src = ev["source"]["issue"]
                pr_author = (src.get("user") or {}).get("login")
                events.append({"type": "pull_request", "at": at, "actor": pr_author, "title": src["title"],
                               "url": src["html_url"], "state": "merged" if src.get("pull_request", {}).get("merged_at") else src["state"],
                               "by_reporter": pr_author in mine})
            elif t == "referenced" and ev.get("commit_id"):
                events.append({"type": "commit", "at": at, "actor": actor, "url": f"https://github.com/{owner}/{repo}/commit/{ev['commit_id']}"})
            elif t == "milestoned":
                events.append({"type": "milestoned", "at": at, "actor": actor, "milestone": (ev.get("milestone") or {}).get("title")})
        followed = [(owner, repo, num, False)] if kind == "pull" else []      # a report that is a PR: its comments are read above
        for ev in events:
            if ev["type"] == "pull_request":
                m = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", ev["url"] or "")
                if m:
                    followed.append((m.group(1), m.group(2), m.group(3), True))
        for o, r, n, with_comments in dict.fromkeys(followed):
            events.extend(pr_activity(o, r, n, token, mine, with_comments))
        events.sort(key=lambda e: e.get("at") or "")
        last = max([e.get("at") for e in events if e.get("at")] + [issue.get("closed_at") or ""] + [""])
        state = "open" if issue["state"] == "open" else {"completed": "fixed", "not_planned": "closed (not planned)",
                                                          "duplicate": "closed (duplicate)"}.get(issue.get("state_reason"), "closed")
        if kind == "pull" and issue.get("pull_request", {}).get("merged_at"):
            state = "merged"
        pulse = bool(last) and (now - dt.datetime.fromisoformat(last.replace("Z", "+00:00"))).days <= PULSE_DAYS
        items.append({"project": project, "finding": finding, "repo": f"{owner}/{repo}", "number": int(num), "url": url + anchor,
                      "kind": ("issue" if by_us else "reported by others") if kind == "issues" else kind, "title": issue["title"], "reporter": reporter, "reported": issue["created_at"][:10],
                      "state": state, "closed_at": issue.get("closed_at"), "labels": [l["name"] for l in issue.get("labels", [])],
                      "reactions": issue.get("reactions", {}).get("total_count", 0), "maintainer_events": events,
                      "reporter_comments": own_comments, "community_comments": community, "last_activity": last or None, "pulse": pulse})
    return {"as_of": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "window_days": PULSE_DAYS, "count": len(items),
            "with_project_activity": sum(1 for i in items if i["maintainer_events"]),
            "fixed": sum(1 for i in items if i["state"] in ("fixed", "merged")), "items": items}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="docs/UPSTREAM.md")
    ap.add_argument("--out", default="upstream/status.json")
    ap.add_argument("--reporter", default="singhpratech", help="our GitHub login(s), comma separated; their comments are counted, not shown; their fix pull requests are shown")
    a = ap.parse_args()
    data = build(a.md, os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"), tuple(a.reporter.split(",")))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"{a.out}: {data['count']} reports, {data['with_project_activity']} with project-side activity, {data['fixed']} fixed, as of {data['as_of']}")
