let pert_size_picker = $('#pertpicker');

function checkPertSizes() {
    pert_type = pert_type_select.value;
    fetch('/pert_sizes/' + pert_type).then(function (response) {
        response.json().then(function (data) {
            let optionHTML = '';
            for (let ps of data.pert_sizes) {
                optionHTML += '<option value="' + ps[1] + '">' + ps[0] + '</option>';
            }
            pert_size_picker.html(optionHTML);
            pert_size_picker.selectpicker('refresh');
            pert_size_picker.selectpicker('render');
        });
    });
}

pert_type_select.onclick = checkPertSizes;

$(document).ready(checkPertSizes);
