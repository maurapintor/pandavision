function checkModel() {
    var form_model = new FormData();
    form_model.append('file', $('#uploadmodel').prop('files')[0]);
    var uploadmodelok = $('#uploadmodelok');
    $(function () {
        $.ajax({
            statusCode: {
                405: function () {
                    uploadmodelok.html("");
                }
            },
            method: 'POST',
            type: 'POST',
            url: '/upload/model',
            data: form_model,
            contentType: false,
            cache: false,
            processData: false,
            success: function (data) {
                uploadmodelok.html("&#9989;");
            },
            error:
                function (data) {
                    console.log('Failed model!');
                    uploadmodelok.html("&#10060;");
                },
        })
    });
}

checkModel();

function checkData() {
    var form_data = new FormData();
    form_data.append('file', $('#uploaddata').prop('files')[0]);
    var uploaddataok = $('#uploaddataok');
    $(function () {
        $.ajax({
            statusCode: {
                405: function () {
                    uploaddataok.html("");
                }
            },
            type: 'POST',
            url: '/upload/data',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function (data) {
                console.log('Success data!');
                uploaddataok.html("&#9989;");
            },
            error:
                function (data) {
                    console.log('Failed data!');
                    uploaddataok.html("&#10060;");
                },
        })
    });
}

checkData();