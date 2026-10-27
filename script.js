document.getElementById('prediction-form').addEventListener('submit', function(e) {
    
    e.preventDefault();
  
    const ambient = document.getElementById('ambient_temp').value;
    const moduleTemp = document.getElementById('module_temp').value;
    const irrad = document.getElementById('irradiation').value;

    const requestData = {
        ambient_temperature: parseFloat(ambient),
        module_temperature: parseFloat(moduleTemp),
        irradiation: parseFloat(irrad)
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
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
