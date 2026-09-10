# Intent codebook — SpotifyCares

Written by hand based on training-cluster exemplars from `src/taxonomy.py`
(see `reports/taxonomy_clusters.txt`). The definitions and near-misses are
editorial choices, not cluster labels — clustering surfaced the candidate
groupings, a human wrote the boundaries.

Two clusters in the raw run turned out to be near-pure noise (bare
`@spotifycares` mentions, URL-only tweets, "thanks!" follow-ups) — that's what
motivated the `not_a_support_request` bucket below. Without it, a meaningful
share of the dataset gets force-fit into a real intent it doesn't belong to.

Every message gets exactly one primary `intent`. If a message plausibly fits
two, use the **tie-break rule**: label the intent the customer would consider
unresolved if you ignored it, and record the other as `secondary_intent`.

---

### `playback_or_app_bug`
The app or playback misbehaves: skipping, crashing, freezing, songs cutting
out, lockscreen controls not working, sync issues.
- ✅ *"it will play about 2 songs and cut off wont let me press shuffle play again"*
- ✅ *"the lockscreen controls and out of app pause functionality have not been functional since the latest app update"*
- ⚠️ Near-miss: *"library limit (10k songs) is a pretty big failure of premium streaming"* → this is a **design limitation complaint**, not a bug → `feature_request_or_product_feedback`.

### `login_or_account_access`
Can't log in, password reset, account locked, account hacked, linked
Facebook/Google account issues.
- ✅ *"Why I can't login to my spotify account? The username & password right, but keep back to login again"*
- ✅ *"My Premium account (created with FB) is being hacked and being used right now!"*
- ⚠️ Near-miss: *"an unknown device has somehow connected to my spotify"* — access is not yet lost, but this is unauthorized-access risk, still logs here (not `account_data_or_privacy`, which is reserved for data-removal/GDPR-style requests).

### `billing_or_subscription_charge`
Charged wrong amount, charged after promo/trial should have applied, payment
failed but was still billed, wants to upgrade/downgrade, refund request.
- ✅ *"I was trying to do the promo for the Hulu/student and I got charge 3 times"*
- ✅ *"i paid for my premium account but it still free, i paid yesterday at Oxxo"*
- ⚠️ Near-miss: *"Idk how to upgrade to premium"* — no charge dispute, just a how-to → `how_to_or_usage_question`.

### `content_availability_or_licensing`
A specific song/album/artist is missing, region-locked, or was removed.
- ✅ *"why did you delete btob's complete album?"*
- ✅ *"why aren't these tracks available ... in alaska"*
- ⚠️ Near-miss: *"can you guys please add more mixtapes of X"* — a request for content the brand never had, i.e. a suggestion rather than something that disappeared → `feature_request_or_product_feedback`.

### `feature_request_or_product_feedback`
Wants a feature that doesn't exist (explicit-content filter, Chromecast
support, send-song-to-friend), or is giving product feedback / complaining
about a design decision, not reporting something broken.
- ✅ *"why can't I block explicit songs??"*
- ✅ *"When are you going to support Chromecast devices from the Desktop client?"*
- ⚠️ Near-miss: *"please change that playlist format back its such a struggle to find one song"* — reads like feedback, but if a recent update broke something that worked before, treat as `playback_or_app_bug` (regression) if the message names a specific prior working state; otherwise feedback.

### `device_or_platform_compatibility`
Works on one device/OS but not another; specific device/OS/app-version
context is central to the message (used only when the core issue is *this
device doesn't work*, distinct from a general bug report).
- ✅ *"Android 7.0 and Spotify version 8.4.24.871 ... When I restart the app, the player back to normal"*
- ⚠️ Near-miss: most device-mentioning bug reports still primarily describe a bug → default to `playback_or_app_bug` unless the message's actual complaint is "this doesn't work on my device/platform" as opposed to "this doesn't work."

### `account_data_or_privacy`
Wants an account or the data tied to it removed; objects to data use;
language/region/geolocation settings being ignored.
- ✅ *"@spotifycares my email was used to make an account that I do not want can you help me remove it"*
- ✅ *"why Please honor my language settings get rid of geolocation"*

### `how_to_or_usage_question`
Doesn't report a problem — asks how to do something the product already
supports (offline downloads, playlist export, changing username).
- ✅ *"Once in a while... will the playlist still be there... I'll just have to re-download it to offline?"*
- ✅ *"How can I change profile name and picture?"*

### `other`
A genuine support request that doesn't fit any bucket above (e.g. asking
about a contact number, generic "please help" with no extractable problem
after reading full context).
- ✅ *"why is it so difficult to find a contact number I can call to help with my login issues"* — actually a meta-complaint about support access itself.

### `not_a_support_request`
Not a request at all: praise/thanks, jokes, song requests to no one in
particular, bare mentions, URL-only tweets, rants with no actionable ask.
- ✅ *"@spotifycares Y'all are the best! Thanks!"*
- ✅ *"I need #SUPERSLIMEY on <URL> NOW!!!!"* (a song request shouted at the brand, not a support issue)
- ✅ bare `@spotifycares` mentions and URL-only tweets with no text

This bucket is not optional. In the sampled clusters, roughly 15–20% of
messages addressed to the brand handle were not requests at all. A taxonomy
that force-fits these into a real intent would be measuring noise as signal
everywhere downstream.

---

## Sampling note for the golden set (§7 of the plan)

Golden examples are drawn **only** from `data/interim/SpotifyCares_test_pool.jsonl`
(the latest 20% of the chronological episode split — never part of the retrieval corpus).
Sampling is stratified across the 10 intents above using a first-pass cheap
classifier (TF-IDF nearest centroid against the codebook examples) purely to
stratify candidate selection, not to label. Rare intents
(`account_data_or_privacy`, `device_or_platform_compatibility`) and messages
the stratifier is least confident about are oversampled so the golden set
stresses the system instead of confirming it.

**Labelling itself is a separate step from sampling.** Every candidate is
labelled by hand against this codebook — an intent, an escalation call, and
a one-line reason — with no model output in the loop before the annotator
sees the message. See REPORT.md for the limits of a single-annotator set.

The current pool is a legacy stratified challenge set, not the 150-random /
50-challenge sample proposed in the revised plan. It cannot estimate ordinary
inbound-volume coverage. The source partition is the latest 20%, not a third.
Current extraction uses direct exchanges, so full connected-component isolation
and near-duplicate removal have not been established. Hand labelling does not
repair those sampling limitations.
