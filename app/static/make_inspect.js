const chartsByCanvasId = {};

const destroyChartIfNecessary = (canvasId) => {
    if (chartsByCanvasId[canvasId]) {
        chartsByCanvasId[canvasId].destroy();
    }
}

const registerNewChart = (canvasId, chart) => {
    chartsByCanvasId[canvasId] = chart;
}


// makes curves plots with attack debugging info
function makeInspect(results, plotContent) {
    var ctx = document.getElementById("inspectResults").getContext('2d');
    var data = {
        labels: Array.apply(null, {length: results[plotContent].length}).map(eval.call, Number),
        datasets: [{
            label: '',
            fill: false,
            data: zip(Array.apply(null, {length: results[plotContent].length}).map(eval.call, Number), results[plotContent]),
            borderColor: 'rgb( 35, 128, 126)',
            backgroundColor: 'rgba(35,128,126,0.73)',
            tension: 0.05,
        }]
    }

    let canvasId = 2;
    destroyChartIfNecessary(canvasId);
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
                        callbacks: {
                            label: function (context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += 'accuracy: ' + (context.parsed.y * 100).toFixed(2) + ' %';
                                }
                                return label;
                            },
                            title: function (context) {
                                let title = context.title || '';
                                if (title) {
                                    title += ': ';
                                }
                                if (context.title !== null) {
                                    title += 'epsilon = ' + (context[0].label);
                                }
                                return title;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        suggestedMin: 0,
                        suggestedMax: 1.1,
                    },
                },

            },
        }
    );
    registerNewChart(canvasId, myChart);

}

function getInspectData(eps_idx, sample_idx) {
    var scripts = document.getElementById('inspect');
    var jobID = scripts.getAttribute('jobid');
    var plotOptions = document.getElementsByName('plotOptions')
    for(i = 0; i < plotOptions.length; i++) {
                if(plotOptions[i].checked)
                    var selected = plotOptions[i].value;
            }

    $.ajax({
        url: `/security_evaluations/${jobID}/inspect/${eps_idx}/${sample_idx}`,
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
            getInspectData(1, 2);
        }
    });
}

let plotContentSelect = document.getElementById('plotContentSelect')

plotContentSelect.onchange = getSamples;
$(document).ready(getSamples);
