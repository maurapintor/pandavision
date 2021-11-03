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
            console.log(data)
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

const footer = (tooltipItems) => {
    console.log("fofoooof")
    let sum = 0;

    tooltipItems.forEach(function (tooltipItem) {
        sum += tooltipItem.parsed.y;
    });
    return 'Sum: ' + sum;
};

function makeGraph(results) {
    var ctx = document.getElementById("secEvalOutput").getContext('2d');
    var data = {
        labels: results['sec-curve']['x-values'],
        datasets: [{
            label: '',
            data: results['sec-curve']['y-values'],
            borderColor: 'rgba(0,117,70)',
            backgroundColor: 'rgba(0,117,70,0.44)',
        }]
    }
    var myChart = new Chart(ctx, {
            type: 'line',
            data: data,
            options: {
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                let label = context.dataset.label || '';

                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += 'accuracy: ' + context.dataset.label ;
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        }
    );
}