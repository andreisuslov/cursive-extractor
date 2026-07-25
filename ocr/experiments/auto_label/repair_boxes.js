export const meta = {
  name: 'repair-straddling-boxes',
  description: 'Context-aware judge for accepted boxes flagged as possibly swallowing a neighbouring line',
  phases: [
    { title: 'Judge', detail: 'see box amid neighbours: keep / fix / reject' },
    { title: 'Reverify', detail: 'adversarial slice check on repaired boxes' },
  ],
}

const TOOL = "/private/tmp/claude-503/-Users-ansuslov-Documents-Development-cursivetransformer/62d8d6f6-ca4d-46be-8056-03188d9d4b44/scratchpad/auto/wordtool.py"
const BATCHES = [[{"pg": "1", "i": 9}, {"pg": "1", "i": 18}, {"pg": "1", "i": 27}, {"pg": "1", "i": 36}, {"pg": "1", "i": 40}, {"pg": "1", "i": 98}, {"pg": "1", "i": 102}, {"pg": "1", "i": 104}], [{"pg": "1", "i": 124}, {"pg": "1", "i": 127}, {"pg": "1", "i": 135}, {"pg": "1", "i": 137}, {"pg": "1", "i": 140}, {"pg": "1", "i": 144}, {"pg": "1", "i": 152}, {"pg": "1", "i": 154}], [{"pg": "1", "i": 162}, {"pg": "1", "i": 170}, {"pg": "1", "i": 172}, {"pg": "1", "i": 173}, {"pg": "1", "i": 182}, {"pg": "1", "i": 184}, {"pg": "1", "i": 187}, {"pg": "1", "i": 191}], [{"pg": "1", "i": 192}, {"pg": "1", "i": 193}, {"pg": "1", "i": 194}, {"pg": "1", "i": 199}, {"pg": "1", "i": 211}, {"pg": "1", "i": 212}, {"pg": "1", "i": 213}, {"pg": "1", "i": 221}], [{"pg": "1", "i": 222}, {"pg": "1", "i": 232}, {"pg": "2", "i": 23}, {"pg": "2", "i": 45}, {"pg": "2", "i": 62}, {"pg": "2", "i": 68}, {"pg": "2", "i": 75}, {"pg": "2", "i": 79}], [{"pg": "2", "i": 81}, {"pg": "2", "i": 83}, {"pg": "2", "i": 118}, {"pg": "2", "i": 119}, {"pg": "2", "i": 121}, {"pg": "2", "i": 122}, {"pg": "2", "i": 123}, {"pg": "2", "i": 127}], [{"pg": "2", "i": 128}, {"pg": "2", "i": 133}, {"pg": "2", "i": 134}, {"pg": "2", "i": 135}, {"pg": "2", "i": 142}, {"pg": "2", "i": 149}, {"pg": "2", "i": 153}, {"pg": "2", "i": 156}], [{"pg": "2", "i": 157}, {"pg": "2", "i": 159}, {"pg": "2", "i": 175}, {"pg": "2", "i": 177}, {"pg": "2", "i": 183}, {"pg": "2", "i": 185}, {"pg": "2", "i": 187}, {"pg": "2", "i": 193}], [{"pg": "2", "i": 196}, {"pg": "2", "i": 252}, {"pg": "2", "i": 256}, {"pg": "2", "i": 257}, {"pg": "2", "i": 258}, {"pg": "2", "i": 262}, {"pg": "2", "i": 264}, {"pg": "2", "i": 265}], [{"pg": "2", "i": 266}, {"pg": "2", "i": 267}, {"pg": "2", "i": 285}, {"pg": "2", "i": 288}, {"pg": "2", "i": 289}, {"pg": "2", "i": 297}, {"pg": "2", "i": 300}, {"pg": "2", "i": 302}], [{"pg": "2", "i": 305}, {"pg": "2", "i": 312}, {"pg": "2", "i": 315}, {"pg": "2", "i": 318}, {"pg": "2", "i": 338}, {"pg": "2", "i": 340}, {"pg": "2", "i": 346}, {"pg": "2", "i": 350}], [{"pg": "2", "i": 353}, {"pg": "2", "i": 358}, {"pg": "2", "i": 381}, {"pg": "2", "i": 382}, {"pg": "2", "i": 385}, {"pg": "2", "i": 393}, {"pg": "2", "i": 396}, {"pg": "2", "i": 397}], [{"pg": "2", "i": 398}, {"pg": "2", "i": 400}, {"pg": "2", "i": 401}, {"pg": "2", "i": 408}, {"pg": "2", "i": 420}, {"pg": "2", "i": 421}, {"pg": "2", "i": 425}, {"pg": "2", "i": 427}], [{"pg": "2", "i": 428}, {"pg": "2", "i": 452}, {"pg": "2", "i": 453}, {"pg": "2", "i": 455}, {"pg": "2", "i": 456}]]

const JUDGE_SCHEMA = {
  type: 'object', required: ['words'],
  properties: { words: { type: 'array', items: {
    type: 'object', required: ['pg', 'i', 'verdict'],
    properties: {
      pg: { type: 'string' }, i: { type: 'integer' },
      verdict: { enum: ['keep', 'fix', 'reject'] },
      box: { type: 'array', items: { type: 'number' } },
      cuts: { type: 'array', items: { type: 'number' } },
      note: { type: 'string' },
    } } } },
}
const VERIFY_SCHEMA = {
  type: 'object', required: ['words'],
  properties: { words: { type: 'array', items: {
    type: 'object', required: ['pg', 'i', 'pass'],
    properties: { pg: { type: 'string' }, i: { type: 'integer' }, pass: { type: 'boolean' },
      bad_slices: { type: 'array', items: { type: 'integer' } }, note: { type: 'string' } },
  } } },
}

const judgePrompt = (jobs) => `These word boxes were auto-accepted, but a geometric pre-filter says each MIGHT swallow part of a neighbouring text line. Judge each one in full page context. Repo root /Users/ansuslov/Documents/Development/cursivetransformer.

Render (Bash), then Read the PNG:
  WORDTOOL_PAGE=<pg> python3 ${TOOL} ctx <i>
GREEN rectangle = the box under judgement. BLUE rectangles = other detected words. It prints the box, its height, and the page median box height.
You may also render a candidate correction: WORDTOOL_PAGE=<pg> python3 ${TOOL} ctx <i> --box X0,Y0,X1,Y1
And to read the ink closely: WORDTOOL_PAGE=<pg> python3 ${TOOL} crop <i> --box X0,Y0,X1,Y1   (green lines = box left/right edges, red ruler = box-local x in px)

Words to judge: ${JSON.stringify(jobs)}

For each: does the GREEN box contain its whole expected word AND ONLY that word's line?
- 'keep'   — box is fine (a tall box is FINE if the extra height is just its own ascenders/descenders and it holds no other word's body).
- 'fix'    — you can tighten it to exactly its own word/line: return the corrected box in PAGE coords AND fresh cuts (box-local x, exactly len(text)-1, ascending, strictly inside the box) for the corrected box. Confirm with a re-render before returning.
- 'reject' — it contains a neighbouring word's body, or the target word is clipped/absent and you cannot fix it in 2 attempts.
Prefer 'reject' over a doubtful 'fix'. Return ALL ${jobs.length} words.`

const verifyPrompt = (words) => `Adversarial re-verification of REPAIRED cursive letter cuts. Repo root /Users/ansuslov/Documents/Development/cursivetransformer.

For each word run then Read:
  WORDTOOL_PAGE=<pg> python3 ${TOOL} slices <i> --box X0,Y0,X1,Y1 --cuts c1,c2,...
Top = the crop with red cut lines; bottom = each slice tile captioned with the letter it would be labelled.

Words: ${JSON.stringify(words)}

Judge every slice: does its ink read as the captioned letter (as a cursive fragment)? Ligature entry/exit strokes are fine. FAIL a slice that holds most of a neighbouring letter's body, is empty (for a letter that is not i/l/t), shows ink from another text line, or is simply the wrong letter. DEFAULT TO FAIL WHEN UNCERTAIN — a wrong label poisons training data.
pass=true only if EVERY slice passes. Return ALL ${words.length} words.`

const out = await pipeline(
  BATCHES,
  (jobs, _o, bi) => agent(judgePrompt(jobs), { label: `judge:b${bi}`, phase: 'Judge', schema: JUDGE_SCHEMA }),
  (judged, jobs, bi) => {
    if (!judged) return null
    const fixed = judged.words.filter(w => w.verdict === 'fix' && w.box && w.cuts && w.cuts.length)
    if (!fixed.length) return { judged: judged.words, verified: [] }
    return agent(verifyPrompt(fixed.map(w => ({ pg: w.pg, i: w.i, box: w.box, cuts: w.cuts }))),
      { label: `reverify:b${bi}`, phase: 'Reverify', schema: VERIFY_SCHEMA })
      .then(v => ({ judged: judged.words, verified: v ? v.words : [] }))
  },
)

const ok = out.filter(Boolean)
const judged = ok.flatMap(r => r.judged)
const verified = ok.flatMap(r => r.verified)
return { judged, verified,
  counts: judged.reduce((m, w) => (m[w.verdict] = (m[w.verdict] || 0) + 1, m), {}),
  reverify_pass: verified.filter(v => v.pass).length }
