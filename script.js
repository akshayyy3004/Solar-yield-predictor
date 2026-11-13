const modelData = {
    gb: { r2: "88.42%", mae: "152.4 kWh", scatter: "static/gb_scatter.png", curve: "static/gb_learning_curve.png" },
    rf: { r2: "85.12%", mae: "189.6 kWh", scatter: "static/rf_scatter.png", curve: "static/rf_learning_curve.png" },
    svr: { r2: "72.30%", mae: "310.2 kWh", scatter: "static/svr_scatter.png", curve: "static/svr_learning_curve.png" }
};

document.getElementById('model-selector').addEventListener('change', function(e) {
    const selectedModel = e.target.value; 
    
    document.getElementById('r2-score').innerText = modelData[selectedModel].r2;
    document.getElementById('mae-score').innerText = modelData[selectedModel].mae;
    
    // Swap BOTH graphs
    document.getElementById('scatter-graph').src = modelData[selectedModel].scatter;
    document.getElementById('learning-curve-graph').src = modelData[selectedModel].curve;
});

document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const requestData = {
        ambient_temperature: parseFloat(document.getElementById('ambient_temp').value),
        module_temperature: parseFloat(document.getElementById('module_temp').value),
        irradiation: parseFloat(document.getElementById('irradiation').value),
        model_type: document.getElementById('model-selector').value 
    };

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        if(data.status === 'success') {
            document.getElementById('yield-result').innerText = data.predicted_yield_kwh + " kWh";
            document.getElementById('result-box').style.display = 'block';
        } else {
            alert("Error: " + data.message);
        }
    })
    .catch(error => console.error('Error:', error));
});
