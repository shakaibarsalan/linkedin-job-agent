# What went wrong, with numbers

The first version of this project drove LinkedIn's Easy Apply in a real
browser, logged in as me. On 29 July 2026 LinkedIn sent an unusual-activity
notice. I stopped everything that could touch LinkedIn, destroyed the stored
session, and wrote this down so I would not do it again.

The browser half is gone from this repo. What is left is the half that never
touches LinkedIn. This is why.

---

## 1. What LinkedIn said

> Over time, our systems have shown your account has accessed a high volume of
> LinkedIn profile data. This often happens because of a third-party tool or
> browser extension... These tools can move LinkedIn data to outside services
> without your knowledge.

No restriction was applied. The account remains in good standing. It should be
treated as **the only warning that will be given**.

## 2. What actually caused it — with numbers

On **28 July alone**, the account made **65 authenticated page loads** from the
worker, plus roughly **10 more** from ad-hoc diagnostic scripts. About **75 in
a single day**, on a session captured two days earlier.

| Category | Loads | Applications produced |
|---|---:|---:|
| Feed loads (session verification) | 31 | 0 |
| Job pages | 29 | 0 |
| Search pages | 8 | 0 |
| Diagnostic scripts (outside the worker) | ~10 | 0 |

Zero applications were ever submitted. Every one of those loads was spent
finding out why the code did not work yet.

LinkedIn's wording — "profile data" — is accurate. Job pages carry *People you
can reach out to*, *Meet the hiring team* and alumni counts, so reading 29 job
pages and 8 search pages pulled a large amount of member data as a side effect.

### It was not the n8n workflows

The execution list looks alarming — "Approve from the Sheet" ran every ten
minutes for a day, ~144 executions. **It never contacted LinkedIn.** Nor did
any other workflow:

| Workflow | Talks to |
|---|---|
| 06 Approve from the Sheet | Google Sheets, our own API |
| 05 Apify Sourcing | Apify's servers (they scrape, not us) |
| 04 Google Sheets Sync | Our API, then Google |
| 01 Daily Digest | Our API, then Telegram |

Unpublishing them was the right instinct, but they were not the cause.

## 3. Every mistake, and how to prevent it

### 3.1 Development happened against the live account — the root cause

LinkedIn had rebuilt the job page with hashed CSS class names
(`_0a89876f _92fbef6a …`), so every selector in the project was dead. That was
discovered by **repeatedly loading real pages to see what broke**. Each guess
cost a real page load on a real account.

**Prevention.** Save one copy of each page's HTML to a fixture file and iterate
against that offline. Selector work needs the *shape* of the page, not a live
connection. One live page load produces a fixture that supports a hundred
iterations. This alone would have removed roughly 90% of the traffic.

### 3.2 Session verification loaded the heaviest page on the site — 31 times

Before every batch, the worker loaded the **entire LinkedIn feed** just to
confirm the cookie still worked. It produced nothing, and it was the single
largest category of traffic.

**Prevention.** Verify the session as a side effect of the first page you
genuinely need, rather than as a separate step. If a dedicated check is really
wanted, it should be the lightest authenticated page available, and cached for
hours rather than repeated per batch.

### 3.3 Four runaway loops, each discovered by running it

* a failed submission batch retried every ~70 seconds, opening a browser each time
* an unreadable application retried with no attempt cap
* any exception escaping a worker tick caused an immediate retry
* a refusal to apply *outside active hours* still opened a browser and loaded
  the feed first — **eleven times in three minutes**

That last one is the clearest illustration: the control meant to keep the
account quiet at night was the loudest thing running.

**Prevention.** Every retry path needs a cap and a backoff *before* it is ever
pointed at a live service. These were all found in production because they were
never exercised anywhere else.

### 3.4 The safety ceiling was written after the traffic it was meant to prevent

`MAX_PAGE_LOADS_PER_HOUR` (a hard cap across all code paths) was built on the
night of 28 July — **after** the traffic that triggered the warning. It would
have stopped this.

**Prevention.** The budget goes in before the first live request, not after the
fourth bug. Brakes are not a later refinement.

### 3.5 Diagnostic scripts bypassed every control

The `diag_*.py` and `verify_*.py` scripts called the browser directly: no rate
limit, no delays, no counting. Even the guard later added would not have caught
them, because they do not pass through it.

**Prevention.** There should be exactly one way to reach LinkedIn in the whole
codebase, and it should be impossible to write a script that avoids it. A
safety net with a documented bypass is not a safety net.

### 3.6 The account had no history to absorb it

The session was captured on 26 July and driven hard on 28 July. A brand-new
session that immediately pulls job pages at machine pace, with no months of
ordinary human browsing behind it, is the most conspicuous possible pattern.

**Prevention.** Nothing to fix in code — this is an argument for the account
being used normally, by a human, for a long time before any automation touches
it, and for the automation's volume never standing out against that baseline.

## 4. "Should we use a test account?"

**The instinct is exactly right. The specific implementation is dangerous.**

Right: *never develop against the account you cannot afford to lose.* That is
the correct lesson and it is the main one.

But a second LinkedIn account is the wrong way to act on it:

* LinkedIn's User Agreement requires **one account per real person**. A test
  account is itself a violation, and they are good at detecting them.
* Accounts are linked by **device, browser fingerprint and IP**. A test account
  created on the same laptop and connection, then banned for automation, can
  associate that ban with your real account. **You could lose the real one
  trying to protect it.**
* A fresh test account also has no history, so it behaves nothing like the
  target environment. It would get flagged faster, teaching you little.

**What to do instead — saved HTML fixtures.**

Almost everything that broke here was selector work: where is the description,
where is the Apply button, what does the form look like. None of that needs a
live connection. Save the page once, iterate against the file forever. No
account, no risk, and faster besides — no network round trip per attempt.

That covers roughly 95% of development. The remaining 5% — "does clicking the
real Submit button actually work" — genuinely needs a real account, once, with
every safety control already in place and verified.

## 5. Is this automation possible at all?

Split it in two, because the two halves have completely different risk.

### The safe half — already built and working

Finding jobs, scoring them against your CV, drafting cover letters, checking
those letters for claims your CV does not support, and putting the results in a
spreadsheet. **This never touches LinkedIn.** Apify scrapes from their own
servers; everything after that is our API and Google Sheets.

This half was working. On 28 July it sourced 78 jobs, scored all of them, wrote
letters, and correctly refused a role wanting twelve years of experience. It
carries no account risk whatsoever.

### The risky half — pressing Submit

Technically possible; we got the form to page 3 of 4. But:

* LinkedIn's User Agreement prohibits automated access, full stop. There is no
  API or developer mode that permits it — that was checked, and none of their
  published APIs cover applying to jobs as yourself.
* You have now had a first warning. A second event could mean permanent
  restriction of an account that is part of your professional identity.
* The honest cost/benefit: automating the final click saves perhaps **two
  minutes per application**. The downside is your professional presence.

### What is actually recommended

**Keep the safe half. Do the last click yourself.**

Each morning the system tells you: here are four jobs worth applying to, scored,
with a cover letter already drafted and fact-checked against your CV. You open
each one, read the letter, paste it, and press Apply.

That is where nearly all the value was. The scoring, the drafting and the
honesty check are the hard parts and the parts that save real time — deciding
whether a job is worth your effort, and writing something truthful about
yourself for it. Pressing a button was never the expensive step.

It also removes the risk entirely, rather than managing it.

## 6. If the browser half ever comes back

In this order, and not out of it:

1. **Restart only the safe half.** Apify → score → cover letter → sheet. No
   browser, no session, no risk. It was working.
2. **Do the applying by hand** for at least a few weeks. See whether the last
   click is genuinely worth automating. It probably is not.
3. If it still seems worth it: **build against fixtures**, get every safety
   control verified first, then use the real account exactly once to confirm —
   with `MAX_PAGE_LOADS_PER_HOUR` around **10**, not the 120 that was sized for
   development.
4. Never let a diagnostic script reach LinkedIn outside the one guarded path.

The final rule, which is the whole lesson:

> **The live account is for the finished thing. Never for finding out whether
> it works.**
