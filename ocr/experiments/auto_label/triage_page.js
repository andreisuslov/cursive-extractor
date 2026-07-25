export const meta = {
  name: 'auto-label-page002',
  description: 'No-human labelling attempt: box QA + cut proposal + adversarial slice verify on all 460 words of page_002',
  phases: [
    { title: 'TriageCut', detail: '46 agents x ~10 words: verify/fix box, propose letter cuts' },
    { title: 'Verify', detail: 'adversarial per-slice check: does slice k read as letter k?' },
    { title: 'Package', detail: 'write labels_auto.json (done=true only for verified)' },
  ],
}

const TOOL = '-m ocr.experiments.auto_label.render'
const OUT = '/private/tmp/claude-503/-Users-ansuslov-Documents-Development-cursivetransformer/62d8d6f6-ca4d-46be-8056-03188d9d4b44/scratchpad/auto'
const N = 460, B = 10
const batches = []
for (let s = 0; s < N; s += B) batches.push(Array.from({length: Math.min(B, N - s)}, (_, k) => s + k))

const TRIAGE_SCHEMA = {
  type: 'object', required: ['words'],
  properties: { words: { type: 'array', items: {
    type: 'object', required: ['i', 'status', 'box', 'cuts', 'conf'],
    properties: {
      i: { type: 'integer' },
      status: { enum: ['ok', 'fixed', 'clipped', 'mismatch', 'unreadable'] },
      box: { type: 'array', items: { type: 'number' }, minItems: 4, maxItems: 4 },
      cuts: { type: 'array', items: { type: 'number' } },
      conf: { type: 'number' }, note: { type: 'string' },
    } } } },
}
const VERIFY_SCHEMA = {
  type: 'object', required: ['words'],
  properties: { words: { type: 'array', items: {
    type: 'object', required: ['i', 'pass', 'bad_slices'],
    properties: { i: { type: 'integer' }, pass: { type: 'boolean' },
      bad_slices: { type: 'array', items: { type: 'integer' } }, note: { type: 'string' } },
  } } },
}

const triagePrompt = (idxs) => `You are labelling cursive handwriting word boxes WITHOUT a human. Work from repo root /Users/ansuslov/Documents/Development/cursivetransformer.

Helper (run via Bash, then Read the produced PNG):
  WORDTOOL_PAGE=2 python3 ${TOOL} crop I                       -> renders word I's box + 40% context. GREEN lines = box edges. RED ruler = BOX-LOCAL x in original px (0 = box left; ink left of 0 is outside the box). Prints JSON with the word's expected text and box (page coords).
  WORDTOOL_PAGE=2 python3 ${TOOL} crop I --box X0,Y0,X1,Y1     -> same but with an adjusted box (page coords).

For EACH word index in ${JSON.stringify(idxs)}:
1. Render + Read the crop. Decide if the box contains its expected word FULLY (whole word inside green lines; small overhang of ascenders/descenders is fine).
2. If the box is shifted/clipped/wrong: compute a corrected box in PAGE coords (use the box-local ruler: e.g. word actually starts at local x=60 and box_w=150 -> new x0 = old x0+60; extend x1 similarly; you may also shift y). Re-render with --box to confirm. Up to 2 correction attempts. status='fixed' if you fixed it, 'clipped'/'mismatch' if you could not.
3. For words whose final box is good (status ok/fixed): propose letter-boundary cuts as BOX-LOCAL x positions (relative to the FINAL box's left edge), exactly len(text)-1 cuts, ascending, each strictly between 0 and box width. Place each cut at the ink gap / ligature between consecutive letters. For 1-letter words, cuts=[].
4. conf: your 0-1 confidence in the cuts. note: one short phrase (what you saw / what was wrong).
For status mismatch/unreadable/clipped return cuts=[] and the best box you found.
Be honest: if you cannot actually read the word in the crop, that is 'unreadable', not low-conf cuts.
Return ALL ${idxs.length} words in the schema.`

const verifyPrompt = (words) => `Adversarial verification of cursive letter cuts. Repo root /Users/ansuslov/Documents/Development/cursivetransformer.

For EACH word below, run via Bash then Read the PNG:
  WORDTOOL_PAGE=2 python3 ${TOOL} slices I --box X0,Y0,X1,Y1 --cuts c1,c2,...
The image shows the word crop with red cut lines (top) and each slice as a tile captioned with the letter it would be labelled as (bottom).

Words to verify (i, box page-coords, cuts box-local, expected text is printed by the tool):
${JSON.stringify(words)}

For each slice k judge: does the ink in tile k plausibly read as its captioned letter, as a cursive fragment? Entry/exit ligature strokes are fine; a slice containing most of a NEIGHBOURING letter's body, an empty slice for a non-i/l letter, or a slice with the wrong letter entirely is a FAIL. DEFAULT TO FAIL WHEN UNCERTAIN — a wrong accepted label poisons training data; a rejected good label just goes back to a human.
pass=true only if EVERY slice passes. bad_slices = 0-based indices of failing slices.
Return ALL words in the schema.`

log(`Triaging ${N} words in ${batches.length} batches of ${B}`)

const results = await pipeline(
  batches,
  (idxs, _o, bi) => agent(triagePrompt(idxs), { label: `triage:${idxs[0]}-${idxs[idxs.length-1]}`, phase: 'TriageCut', schema: TRIAGE_SCHEMA }),
  (triage, idxs, bi) => {
    if (!triage) return null
    const good = triage.words.filter(w => (w.status === 'ok' || w.status === 'fixed') && w.cuts)
    const toVerify = good.filter(w => w.cuts.length > 0).map(w => ({ i: w.i, box: w.box, cuts: w.cuts }))
    if (!toVerify.length) return { triage: triage.words, verify: [] }
    return agent(verifyPrompt(toVerify), { label: `verify:batch${bi}`, phase: 'Verify', schema: VERIFY_SCHEMA })
      .then(v => ({ triage: triage.words, verify: v ? v.words : [] }))
  },
)

const all = results.filter(Boolean)
const triaged = all.flatMap(r => r.triage)
const verdicts = new Map(all.flatMap(r => r.verify).map(v => [v.i, v]))
const accepted = triaged.filter(w => verdicts.get(w.i)?.pass)
  .map(w => ({ i: w.i, box: w.box.map(Math.round), cuts: w.cuts }))
const stats = {
  total: triaged.length,
  by_status: triaged.reduce((m, w) => (m[w.status] = (m[w.status] || 0) + 1, m), {}),
  verified_pass: accepted.length,
  verified_fail: [...verdicts.values()].filter(v => !v.pass).length,
}
log(`Triage done: ${JSON.stringify(stats.by_status)}; verified pass ${stats.verified_pass}`)

phase('Package')
const pkg = await agent(`Write the auto-label output for PAGE 2. Repo root /Users/ansuslov/Documents/Development/cursivetransformer.

1. Write ${OUT}/accepted_p2.json containing exactly this JSON: ${JSON.stringify(accepted)}
2. Write ${OUT}/triage_full_p2.json containing exactly this JSON: ${JSON.stringify({ triaged, verdicts: [...verdicts.values()], stats })}
3. Run a small python script that does the following:
   - Load outputs/0-02-1977-yellow-spiral-bound-notebook_pages_8-14/page_002/boxes_refined.json (list of {text, box_2d}); convert each box_2d [ymin,xmin,ymax,xmax] (0-1000 scale) to page px using the page image size (2400x1800): x0=round(xmin/1000*2400), y0=round(ymin/1000*1800), x1=round(xmax/1000*2400), y1=round(ymax/1000*1800).
   - Load ${OUT}/triage_full_p2.json and ${OUT}/accepted_p2.json.
   - Build a list of 460 portal words in order: {"text": <text>, "box": <triaged final box if that index appears in the triage records, else the converted box>, "cuts": [], "erase": [], "done": false}. For each entry in accepted_p2.json set that index's box to the accepted box, cuts to [[[x,0],[x,H]],...] polylines from the accepted cut x-positions (H = box[3]-box[1]), and done to true.
   - Save as outputs/0-02-1977-yellow-spiral-bound-notebook_pages_8-14/page_002/labels_auto.json
   - Also save outputs/0-02-1977-yellow-spiral-bound-notebook_pages_8-14/page_002/auto_labelled_doc.json as {"page": "<data:image/jpeg;base64,...>", "words": <ONLY the done=true words>} where the data-url is the page PNG (outputs/.../page_002/0-02-1977-yellow-spiral-bound-notebook_pages_8-14_p002_page.png) re-encoded as JPEG quality 85.
   - Print the count of done=true entries.
Return JSON {"written": [paths], "n_done": <count>}.`,
  { label: 'package', phase: 'Package', schema: { type: 'object', required: ['written', 'n_done'], properties: { written: { type: 'array', items: { type: 'string' } }, n_done: { type: 'integer' } } } })

return { stats, n_done: pkg ? pkg.n_done : 0, accepted_indices: accepted.map(a => a.i), files: pkg ? pkg.written : [] }