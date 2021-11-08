/**
 * Simple long polling client based on JQuery
 * https://github.com/sigilioso/long_polling_example/blob/master/static/poll.js
 */

/**
 * Request an update to the server and once it has answered, then update
 * the content and request again.
 * The server is supposed to response when a change has been made on data.
 */
function update(jobId) {
    $.ajax({
        url: `/security_evaluations/${jobId}`,
        success: function (data) {
            switch (data['task_status']) {
                case "finished":
                    $('#spinner').hide();
                    $('#waitText').text("");
                    makeGraph(data['results']);
                    break;
                case "started":
                    $('#waitText').text("Job started...");
                    $('#spinner').show();
                    setTimeout(function () {
                        update(jobId);
                    }, 1000);
                    break;
                case "queued":
                    $('#waitText').text("Please wait ...");
                    $('#spinner').show();
                    setTimeout(function () {
                        update(jobId);
                    }, 1000);
                    break;
                case "failed":
                    $('#waitText').text("Job Failed.");
                    $('#spinner').hide();
                    break
            }

        }
    });
}


$(document).ready(function () {
    var scripts = document.getElementById('polling');
    var jobID = scripts.getAttribute('jobid');
    update(jobID);
});

const zip = (a, b) => a.map((k, i) => [k, b[i]]);

function makeGraph(results) {
    var ctx = document.getElementById("secEvalOutput").getContext('2d');
    var data = {
        labels: results['sec-curve']['x-values'],
        datasets: [{
            label: '',
            fill: true,
            data: zip(results['sec-curve']['x-values'], results['sec-curve']['y-values']),
            borderColor: 'rgb( 35, 128, 126)',
            backgroundColor: 'rgba(35,128,126,0.73)',
            tension: 0.3,
        }]
    }
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
    $('#accText').css("visibility",'visible');
    $('#pertSizeText').css("visibility",'visible');
}

