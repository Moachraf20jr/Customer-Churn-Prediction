const express = require('express')
const path = require('path')
const { spawn } = require('child_process')
const fs = require('fs')

const app = express()
const PORT = process.env.PORT || 3000

app.use(express.json())
app.use(express.static(path.join(__dirname, 'public')))
app.use('/results', express.static(path.join(__dirname, 'results')))

const METADATA_PATH = path.join(__dirname, 'models', 'metadata.pkl')
const FEATURE_IMPORTANCE_PATH = path.join(__dirname, 'results', 'metrics', 'feature_importance_Random_Forest.csv')

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' })
})

app.get('/api/model-info', async (req, res) => {
  try {
    const script = `
import json, joblib
m = joblib.load(r'${METADATA_PATH.replace(/\\/g, '\\\\')}')
print(json.dumps({
  'model_name': m['model_name'],
  'feature_names': m['feature_names'],
  'metrics': m['metrics'],
  'random_state': m['random_state']
}))`
    const meta = await runPython(script)
    let imp = []
    if (fs.existsSync(FEATURE_IMPORTANCE_PATH)) {
      const impScript = `
import csv, json
with open(r'${FEATURE_IMPORTANCE_PATH.replace(/\\/g, '\\\\')}') as f:
  rows = list(csv.DictReader(f))
print(json.dumps(rows))`
      imp = await runPython(impScript)
    }
    res.json({ ...meta, feature_importance: imp })
  } catch (err) {
    console.log('model info error:', err.message)
    res.status(500).json({ error: err.message })
  }
})

function runPython(code) {
  return new Promise((resolve, reject) => {
    const proc = spawn('python', ['-c', code])
    let out = '', err = ''
    proc.stdout.on('data', d => out += d)
    proc.stderr.on('data', d => err += d)
    proc.on('close', code => {
      if (code !== 0) return reject(new Error(err.trim() || 'Python error'))
      try { resolve(JSON.parse(out.trim())) }
      catch { resolve(out.trim()) }
    })
    proc.on('error', e => reject(new Error('Failed to run Python: ' + e.message)))
  })
}

function predict(data) {
  return new Promise((resolve, reject) => {
    const proc = spawn('python', [path.join(__dirname, 'predict_service.py')])
    let out = '', err = ''
    proc.stdout.on('data', d => out += d)
    proc.stderr.on('data', d => err += d)
    proc.on('close', code => {
      if (code !== 0) {
        let msg = err.trim() || 'prediction failed'
        try { const o = JSON.parse(out); if (o.error) msg = o.error } catch (e) {}
        return reject(new Error(msg))
      }
      try { resolve(JSON.parse(out)) }
      catch (e) { reject(new Error('Failed to parse: ' + e.message)) }
    })
    proc.on('error', e => reject(new Error('Failed to start Python: ' + e.message)))
    proc.stdin.write(JSON.stringify(data))
    proc.stdin.end()
  })
}

const FIELDS = ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
  'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction']

app.post('/api/predict', async (req, res) => {
  try {
    const body = req.body
    if (!Array.isArray(body)) return res.status(400).json({ error: 'Body must be an array' })
    for (let i = 0; i < body.length; i++) {
      for (const f of FIELDS) {
        if (body[i][f] === undefined || body[i][f] === null || body[i][f] === '') {
          return res.status(400).json({ error: `Item ${i}: missing '${f}'` })
        }
      }
    }
    const result = await predict(body)
    if (result.error) return res.status(500).json({ error: result.error })
    res.json({ predictions: result })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.listen(PORT, () => {
  console.log('server running on http://localhost:' + PORT)
  console.log('press ctrl+c to stop')
})
