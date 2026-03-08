import { useState } from "react";
import axios from "axios";
import "./App.css";
import Papa from "papaparse";

function App() {
  const [dimension, setDimension] = useState(30);
  const [minVal, setMinVal] = useState(-100);
  const [maxVal, setMaxVal] = useState(100);
  const [vector, setVector] = useState("");
  const [selectedFunction, setSelectedFunction] = useState("F1");
  const [fitness, setFitness] = useState(null);
  const [best, setBest] = useState(null);
  const [image, setImage] = useState(null);
  const [worst, setWorst] = useState(null);
  const [loading, setLoading] = useState(false);
  const [size, setSize] = useState(30);
  const [csvData, setCsvData] = useState([]); // { rows: string[][], fileName: string }
  const [csvFileName, setCsvFileName] = useState("");

   // ── Multiple Runs state ──
  const [numRuns, setNumRuns] = useState(15);
  const [numIter, setNumIter] = useState(20);
  const [c1, setC1] = useState(1.5);
  const [c2, setC2] = useState(1.5);
  const [runsLoading, setRunsLoading] = useState(false);
  const [iterLoading, setIterLoading] = useState(false);
  const [runsResult, setRunsResult] = useState(null); // { best, worst, mean, std, image_url }
  const [iterResult, setIterResult] = useState(null); 
  const API = "http://127.0.0.1:5000";

  const generateSolution = async () => {
    try {
      setLoading(true);
      const response = await axios.post(`${API}/generate`, {
        dimension,
        min: minVal,
        max: maxVal,
      });
      setVector(response.data.join(" "));
      setFitness(null);
    } catch (error) {
      console.error("Error generating solution:", error);
    } finally {
      setLoading(false);
    }
  };

  const generateMultipleVectors = async () => {
    try {
      setLoading(true);
      let allVectors = [];
      for (let i = 0; i < size; i++) {
        const response = await axios.post(`${API}/generate`, {
          dimension,
          min: minVal,
          max: maxVal,
        });
        allVectors.push(response.data);
      }
      const csvContent = allVectors.map((v) => v.join(",")).join("\n");
      const blob = new Blob([csvContent], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "vectors.csv";
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error generating vectors:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleCSVUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setCsvFileName(file.name);

    Papa.parse(file, {
      complete: (result) => {
        // Filter out empty rows
        const rows = result.data.filter((row) =>
          row.some((cell) => cell !== "")
        );
        setCsvData(rows);

        // Also populate the vector textarea with the first row
        const values = [];
        rows.forEach((row) => {
          row.forEach((cell) => {
            if (cell !== "") values.push(cell);
          });
        });
        setVector(values.join(" "));
      },
      error: (error) => {
        console.error("CSV Parsing Error:", error);
      },
    });
  };

  const getVectorsFromCSV = () => {
  return csvData.map(row =>
    row
      .filter(cell => cell !== "")
      .map(cell => parseFloat(cell))
  );
};

  const evaluateSolution = async () => {
    try {
      setLoading(true);
      const response = await axios.post(`${API}/evaluate`, {
        vector: vector.trim().split(/\s+/),
        function: selectedFunction,
      });
      setFitness(response.data);
    } catch (error) {
      console.error("Error evaluating:", error);
    } finally {
      setLoading(false);
    }
  };
  // Évaluer chaque vecteur
const evaluateVectors = async () => {
  if (csvData.length === 0) {
    alert("Please load a CSV file first!");
    return;
  }

  try {
    setLoading(true);
    setBest(null);
    setWorst(null);
    setImage(null);

    const vectors = getVectorsFromCSV();

    const response = await axios.post(`${API}/evaluate-csv`, {
      vectors: vectors,
      function: selectedFunction,
    });

    setBest(response.data.best_value);
    setWorst(response.data.worst_value);
    setImage(response.data.image_url);

  } catch (error) {
    console.error("Error evaluating vectors:", error);
  } finally {
    setLoading(false);
  }
};

 // ── Multiple Runs handler ──
  const evaluateMultipleRuns = async () => {
    try {
      setRunsLoading(true);
      setRunsResult(null);

      const response = await axios.post(`${API}/evaluate-multiple-runs`, {
        function: selectedFunction,
        runs: numRuns,
        size: size,
        dimension: dimension,
      });

      setRunsResult(response.data);
    } catch (error) {
      console.error("Error evaluating multiple runs:", error);
    } finally {
      setRunsLoading(false);
    }
  };

const evaluatePSO = async () => {
  try {
    const response = await axios.post(`${API}/evaluate-PSO`, {
      c1,
      c2,
      numRuns,
      function: selectedFunction,
      numIter,
      size,
      dimension
    });

    const data = response.data;

setIterResult({
  best: data.best_fitness,
  best_position: data.best_position,
  curve: data.curve,
  image_url: data.image_url,
  image_url2: data.image_url2,
  image_url3: data.image_url3,
  image_url4: data.image_url4,
  history: data.history,
  avg_curve: data.avg_curve,
  trajectory: data.trajectory,
  stagnation_iter: data.stagnation_iter,
  // ── Add the new stats fields ──
  stats: data.stats,  // contains: best, worst, mean, std, all_best_fits, num_runs
});

  } catch (error) {
    console.error("Error evaluating PSO:", error);
  }
};

const evaluateMultipleIter = async () => {
  try {
    setIterLoading(true);

    const response = await axios.post("http://localhost:5000/evaluate-PSO", {
      function: selectedFunction,
      numIter: numIter,
      size: size,
      dimension: 2,   // 🔥 adapte si besoin
      c1: c1,
      c2: c2
    });

    const data = response.data;

    setIterResult({
      best: data.best_fitness,
      worst: Math.max(...data.convergence_curve),
      image_url: null   // si tu ajoutes une image plus tard
    });

  } catch (error) {
    console.error("Error evaluating PSO:", error);
  } finally {
    setIterLoading(false);
  }
};



  return (
    <div className="page">
      <h1 className="title">Metaheuristics Optimization Lab</h1>

      <div className="card">
        <div className="inputs-row">
          <div className="input-group">
            <label>Dimension</label>
            <input
              type="number"
              value={dimension}
              onChange={(e) => setDimension(e.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Min</label>
            <input
              type="number"
              value={minVal}
              onChange={(e) => setMinVal(e.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Max</label>
            <input
              type="number"
              value={maxVal}
              onChange={(e) => setMaxVal(e.target.value)}
            />
          </div>
        </div>

        <button className="primary-btn" onClick={generateSolution}>
          {loading ? "Generating..." : "Generate Solution"}
        </button>

        <textarea
          className="vector-area"
          rows="4"
          placeholder="Candidate solution..."
          value={vector}
          onChange={(e) => setVector(e.target.value)}
        />

        <div className="action-row">
          <select
            className="select"
            value={selectedFunction}
            onChange={(e) => setSelectedFunction(e.target.value)}
          >
            <option className="runs-meta" value="F1">F1</option>
            <option className="runs-meta" value="F2">F2</option>
            <option className="runs-meta" value="F5">F5</option>
            <option className="runs-meta" value="F8">F8</option>
            <option className="runs-meta" value="F9">F9</option>
            <option className="runs-meta" value="F11">F11</option>
          </select>

          <button className="secondary-btn" onClick={evaluateSolution}>
            {loading ? "Evaluating..." : "Evaluate"}
          </button>
        </div>

        {fitness !== null && (
          <div className="result-box">Fitness: {fitness}</div>
        )}

        <div className="upload-row">
          <label className="upload-label">
            Load CSV File
            <input
              type="file"
              accept=".csv"
              onChange={handleCSVUpload}
              style={{ display: "none" }}
            />
          </label>
        </div>
      <button className="primary-btn" onClick={generateMultipleVectors}>
          {loading ? "Generating..." : "Generate Vectors of CSV random solutions"}
        </button>
      </div>
    <button className="secondary-btn" onClick={evaluateVectors}>
            {loading ? "Evaluating..." : "Evaluate - csv vectors"}
          </button>
    
        {best !== null && (
          <div className="result-box">best: {best}</div>
        )}
        {worst !== null && (
          <div className="result-box">worst: {worst}</div>
        )}
      {/* CSV Table Display */}
      {csvData.length > 0 && (
        <div className="csv-card">
          <div className="csv-header">
            <h2 className="csv-title">📄 {csvFileName}</h2>
            <span className="csv-meta">
              {csvData.length} rows × {csvData[0].length} columns
            </span>
          </div>

          <div className="csv-table-wrapper">
            <table className="csv-table">
              <thead>
                <tr>
                  <th className="row-index">#</th>
                  {csvData[0].map((_, colIdx) => (
                    <th key={colIdx}>Col {colIdx + 1}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {csvData.map((row, rowIdx) => (
                  <tr key={rowIdx} className={rowIdx % 2 === 0 ? "even" : "odd"}>
                    <td className="row-index">{rowIdx + 1}</td>
                    {row.map((cell, cellIdx) => (
                      <td key={cellIdx} title={cell}>
                        {isNaN(cell) || cell === ""
                          ? cell
                          : parseFloat(cell).toFixed(4)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
     {image && <img src={image} alt="plot" width="800" />}

     {/* ===================== */}
      {/* META HEURISTICS SECTION */}
      {/* ===================== */}
      <div className="card runs-card">
        <h2 className="section-title">🔁 Running Multiple Populations</h2>
        <p className="section-desc">
          Generates <strong>{numRuns}</strong> random populations of size{" "}
          <strong>{size}</strong> and returns aggregated performance metrics
          over all runs.
        </p>

        <div className="inputs-row">
          <div className="input-group">
            <label>Number of Runs</label>
            <input
              type="range"
              min={1}
              max={50}
              value={numRuns}
              onChange={(e) => setNumRuns(Number(e.target.value))}
            />
            <span className="slider-value">{numRuns}</span>
          </div>

          <div className="input-group">
            <label>Population Size</label>
            <input
              type="range"
              min={5}
              max={100}
              value={size}
              onChange={(e) => setSize(Number(e.target.value))}
            />
            <span className="slider-value">{size}</span>
          </div>
        </div>

        <div className="runs-meta">
          Function selected: <strong>{selectedFunction}</strong> — change it
          in the dropdown above ☝️
        </div>

        <button
          className="primary-btn"
          onClick={evaluateMultipleRuns}
          disabled={runsLoading}
        >
          {runsLoading ? "Evaluating runs..." : "Evaluate"}
        </button>

        {runsResult && (
          <div className="runs-results">
            <div className="metrics-grid">
              <div className="metric-card metric-best">
                <span className="metric-label">Best</span>
                <span className="metric-value">
                  {runsResult.best.toFixed(4)}
                </span>
              </div>
              <div className="metric-card metric-worst">
                <span className="metric-label">Worst</span>
                <span className="metric-value">
                  {runsResult.worst.toFixed(4)}
                </span>
              </div>
              <div className="metric-card metric-mean">
                <span className="metric-label">Mean (avg error)</span>
                <span className="metric-value">
                  {runsResult.mean.toFixed(4)}
                </span>
              </div>
              <div className="metric-card metric-std">
                <span className="metric-label">STD</span>
                <span className="metric-value">
                  {runsResult.std.toFixed(4)}
                </span>
              </div>
            </div>

            {runsResult.image_url && (
              <img
                src={`${runsResult.image_url}?t=${Date.now()}`}
                alt="Multiple runs plot"
                width="800"
                className="runs-plot"
              />
            )}
          </div>
        )}
      </div>

     {/* ===================== */}
      {/* META HEURISTICS SECTION */}
      {/* ===================== */}
     <div className="card runs-card">
        <h2 className="section-title">🔁 Running WITH PSO</h2>
      

        <div className="inputs-row">
          <div className="input-group">
            <label>Number of Iterations</label>
            <input
              type="range"
              min={1}
              max={200}
              value={numIter}
              onChange={(e) => setNumIter(Number(e.target.value))}
            />
            <span className="slider-value">{numIter}</span>
          </div>

          <div className="input-group">
            <label>Population Size</label>
            <input
              type="range"
              min={5}
              max={100}
              value={size}
              onChange={(e) => setSize(Number(e.target.value))}
            />
            <span className="slider-value">{size}</span>
          </div>

          <div className="input-group">
            <label>Number of Runs</label>
            <input
              type="range"
              min={1}
              max={80}
              value={numRuns}
              onChange={(e) => setNumRuns(Number(e.target.value))}
            />
            <span className="slider-value">{numRuns}</span>
          </div>
          
          <div className="input-group">
            <label>C1</label>
            <input
              type="range"
              min={0}
              max={20}
              value={c1}
              onChange={(e) => setC1(Number(e.target.value))}
            />
            <span className="slider-value">{c1}</span>
          </div>

            <div className="input-group">
            <label>C2</label>
            <input
              type="range"
              min={0}
              max={20}
              value={c2}
              onChange={(e) => setC2(Number(e.target.value))}
            />
            <span className="slider-value">{c2}</span>
          </div>

        </div>
        
        <div className="runs-meta">
          Function selected: <strong>{selectedFunction}</strong> — change it
          in the dropdown above ☝️
        </div>

        <button
          className="primary-btn"
          onClick={evaluatePSO}
          disabled={iterLoading}
        >
          {iterLoading ? "Evaluating runs..." : "Evaluate"}
        </button>

        {iterResult && (
          <div className="runs-results">
            <div className="metrics-grid">
              <div className="metric-card metric-best">
                <span className="metric-label">Best fit</span>
                <span className="metric-value">
                  {iterResult.stats.best.toFixed(4)}
                </span>
              </div>

      <div className="metric-card metric-best">
        <span className="metric-label">All Best fits</span>
        <span className="metric-value">
          {iterResult.stats?.all_best_fits?.map((v, i) => (
            <div key={i}>Run {i + 1}: {v.toFixed(4)}</div>
          )) ?? "—"}
        </span>
      </div>
              
              <div className="metric-card metric-worst">
                <span className="metric-label">mean</span>
                <span className="metric-value">
                  <div>
                 {iterResult.stats.mean.toFixed(4)}
    
</div>
                </span>
              </div>
<div className="metric-card metric-best">
                <span className="metric-label">Stagnation</span>
                <span className="metric-value">
                  {iterResult.stagnation_iter}
                </span>
              </div>

            </div>

            {iterResult.curve && (
              <div>
               <img
                src={`${iterResult.image_url}?t=${Date.now()}`}
                alt="Iter plot"
                width="500"
                className="iter-plot"
              />
              <img
                src={`${iterResult.image_url2}?t=${Date.now()}`}
                alt="Iter plot"
                width="500"
                className="iter-plot"
              />
              <img
                src={`${iterResult.image_url3}?t=${Date.now()}`}
                alt="Iter plot3"
                width="500"
                className="iter-plot"
              />
                            <img
                src={`${iterResult.image_url4}?t=${Date.now()}`}
                alt="Iter plot4"
                width="500"
                className="iter-plot"
              />
              </div>

              
            )}
          </div>
        )}
      </div>
    </div>
      
  );
}

export default App;
