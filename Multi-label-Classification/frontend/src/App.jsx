import React, {useState} from 'react'

export default function App(){
  const [title, setTitle] = useState('')
  const [overview, setOverview] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleSubmit(e){
    e.preventDefault()
    setLoading(true)
    setResult(null)
    setError(null)
    try{
      const resp = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, overview })
      })
      if(!resp.ok){
        const text = await resp.text()
        throw new Error(text || resp.statusText)
      }
      const data = await resp.json()
      setResult(data)
    }catch(err){
      setError(err.message || String(err))
    }finally{
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Movie Tags Predictor</h1>
        <p>Enter a movie title and/or overview and click Predict.</p>
      </header>

      <main>
        <form onSubmit={handleSubmit} className="form">
          <label>
            Title
            <input value={title} onChange={e=>setTitle(e.target.value)} placeholder="Movie title (optional)" />
          </label>

          <label>
            Overview
            <textarea value={overview} onChange={e=>setOverview(e.target.value)} placeholder="Movie overview" rows={6} />
          </label>

          <button type="submit" disabled={loading || (!title && !overview)}>{loading? 'Predicting...':'Predict'}</button>
        </form>

        <section className="result">
          {error && <div className="error">Error: {error}</div>}
          {result && (
            <div>
              <h2>Predicted Tags</h2>
              <p>{result.data}</p>
              {result.tags ? (
                <ul>
                  {result.tags.map((t, i)=> (
                    <li key={i}>{t}</li>
                  ))}
                </ul>
              ) : (
                <pre>{JSON.stringify(result, null, 2)}</pre>
              )}
            </div>
          )}
        </section>
      </main>

      <footer>
        <small>Frontend expects a POST /predict endpoint returning JSON like <code>{'{ "tags": ["Action","Drama"] }'}</code></small>
      </footer>
    </div>
  )
}
