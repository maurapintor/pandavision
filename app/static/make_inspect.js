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
    console.log(results);
    datasets = [];
    datasets.push(
        {
            label: '',
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
    // first eps value is zero and it is skipped
    const eps_idx = epsSelect.selectedIndex + 1;
    const sample_id = sampleSelect.selectedIndex;

    var scripts = document.getElementById('inspect');
    var jobID = scripts.getAttribute('jobid');
    var plotOptions = document.getElementsByName('plotOptions')
    for (i = 0; i < plotOptions.length; i++) {
        if (plotOptions[i].checked)
            var selected = plotOptions[i].value;
    }
    $.ajax({
        url: `/security_evaluations/${jobID}/inspect/${eps_idx}/${sample_id}`,
        success: function (data) {
            makeInspect(data, selected);
        }
    });
}

function getSamples() {
    var scripts = document.getElementById('inspect');
    var jobID = scripts.getAttribute('jobid');

    $.ajax({
        url: `/security_evaluations/${jobID}/inspect`,
        success: function (data) {
            for (eps in data['epsilon_values']) {
                var e = document.createElement("option");
                e.innerHTML = '<option value="' + eps + '">' + data['epsilon_values'][eps] + '</option>';
                epsSelect.appendChild(e);
            }
            for (sample_id in data['num_samples']){
                var s = document.createElement("option");
                s.innerHTML = '<option value="' + sample_id + '">' + data['num_samples'][sample_id] + '</option>';
                sampleSelect.appendChild(s);
            }
        }
    });
}

function showInspect() {
    $('#inspectResultsDiv').css("visibility", 'visible');
    let plotContentSelect = document.getElementById('plotContentSelect')
    let epsSelect = document.getElementById('epsSelect');
    let sampleSelect = document.getElementById('sampleSelect');

    plotContentSelect.onchange = updateCurves;
    plotContentSelect.onclick = updateCurves;
    epsSelect.onchange = updateCurves;
    sampleSelect.onchange = updateCurves;
}
