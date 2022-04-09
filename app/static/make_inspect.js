const chartsByCanvasId = {};

const getChartIfExists = (canvasId, ctx, data) => {
    if (chartsByCanvasId[canvasId]) {
        myChart = chartsByCanvasId[canvasId];
        myChart.data = data;
        myChart.update();
    } else {
        var myChart = new Chart(ctx, {
                    type: 'scatter',
                    data: data,
                    options: {
                        layout: {
                            padding: 50
                        },
                        scales: {x: { title: { display: true, text: 'iterations' }}},
                        interaction: {
                            mode: 'index',
                            intersect: false,
                        },

                        plugins: {
                            legend: {
                                display: false,
                            },
                            tooltip: {
                                displayColors: false,

                            }
                        },

                    },
                }
            )
        ;
    }
    chartsByCanvasId[canvasId] = myChart;
}


// makes curves plots with attack debugging info
function makeInspect(results, plotContent) {
    var ctx = document.getElementById("inspectResults").getContext('2d');
    datasets = [];
    datasets.push(
        {
            label: 'iterations',
            fill: false,
            data: zip(Array.apply(null, {length: results[plotContent].length}).map(eval.call, Number), results[plotContent]),
            borderColor: 'rgb( 35, 128, 126)',
            backgroundColor: 'rgba(35,128,126,0.73)',
            tension: 0.05,
            showLine: true,
        }
    );

    var data = {
        labels: Array.apply(null, {length: results[plotContent].length}).map(eval.call, Number),
        datasets: datasets,
    }
    let canvasId = 'inspect';
    getChartIfExists(canvasId, ctx, data);
}

function updateCurves() {
    let sample_id = sampleSelect.selectedIndex;
    if (sample_id === -1){
        sample_id = 0;
    }
    var scripts = document.getElementById('inspect');
    var jobID = scripts.getAttribute('jobid');
    var plotOptions = document.getElementsByName('plotOptions')
    for (i = 0; i < plotOptions.length; i++) {
        if (plotOptions[i].checked)
            var selected = plotOptions[i].value;
    }


    $.ajax({
        url: `/security_evaluations/${jobID}/inspect/${sample_id}`,
        success: function (data) {
            makeInspect(data, selected);
        }
    });
}

function updateSample(){
    var scripts = document.getElementById('inspect');
    var jobID = scripts.getAttribute('jobid');
    let sample_id = sampleSelect.selectedIndex;
        if (sample_id === -1){
        sample_id = 0;
    }

    let advImageDisplay = document.getElementById('advExampleImage');
    $.ajax({
        url: `/security_evaluations/${jobID}/inspect/${sample_id}/adv_examples`,
        success: function (data) {
            advImageDisplay.src = data;
        }
    });
    let origSampleDisplay = document.getElementById('OriginalSampleImage');
    $.ajax({
        url: `/security_evaluations/${jobID}/inspect/${sample_id}/orig_samples`,
        success: function (data) {
            origSampleDisplay.src = data;
        }
    });
    let diffSampleDisplay = document.getElementById('DiffImage');
    $.ajax({
        url: `/security_evaluations/${jobID}/inspect/${sample_id}/diff`,
        success: function (data) {
            diffSampleDisplay.src = data;
        }
    });

}

function getSamples() {
    var scripts = document.getElementById('inspect');
    var jobID = scripts.getAttribute('jobid');

    $.ajax({
        url: `/security_evaluations/${jobID}/inspect`,
        success: function (data) {
            for (sample_id in data['num_samples']) {
                var s = document.createElement("option");
                s.innerHTML = '<option value="' + sample_id + '">' + data['num_samples'][sample_id] + '</option>';
                sampleSelect.appendChild(s);
            }
        }
    });
}

function updateCurvesAndSample(){
    updateCurves();
    updateSample();
}

function showInspect() {
    $('#inspectResultsDiv').css("visibility", 'visible');
    let plotContentSelect = document.getElementById('plotContentSelect');
    let sampleSelect = document.getElementById('sampleSelect');
    plotContentSelect.onchange = updateCurves;
    sampleSelect.onchange = updateCurvesAndSample;
}
