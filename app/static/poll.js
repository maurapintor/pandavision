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
                    makeGraph(data['data']);
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
            }

        }
    });
}


$(document).ready(function () {
    var scripts = document.getElementById('polling');
    var jobID = scripts.getAttribute('jobid');
    update(jobID);
});

function makeGraph(results) {
    var ctx = document.getElementById("secEvalOutput").getContext('2d');
    var myChart = new Chart(ctx, {
        type: 'horizontalBar',
        data: {
            labels: [results[0][0], results[1][0], results[2][0], results[3][0], results[4][0]],
            datasets: [{
                label: 'Output scores',
                data: [results[0][1], results[1][1], results[2][1], results[3][1], results[4][1]],
                backgroundColor: [
                    'rgba(26,74,4,0.8)',
                    'rgba(117,0,20,0.8)',
                    'rgba(121,87,3,0.8)',
                    'rgba(6,33,108,0.8)',
                    'rgba(63,3,85,0.8)',
                ],
                borderColor: [
                    'rgba(26,74,4)',
                    'rgba(117,0,20)',
                    'rgba(121,87,3)',
                    'rgba(6,33,108)',
                    'rgba(63,3,85)',
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }]
            }
        }
    });
}