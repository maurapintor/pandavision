let pert_type_select = document.getElementById('pert_type')
let attack_select = document.getElementById('attack')
let pert_size_picker = $('#pertpicker');

function checkAttacks() {
    pert_type = pert_type_select.value;
    fetch('/attacks/' + pert_type).then(function (response) {
        response.json().then(function (data) {
            let optionHTML = '';

            for (let attack of data.attacks) {
                optionHTML += '<option value="' + attack[1] + '">' + attack[0] + '</option>';
            }
            attack_select.innerHTML = optionHTML;
        });
    });
}


function checkPertSizes() {
    pert_type = pert_type_select.value;
    fetch('/pert_sizes/' + pert_type).then(function (response) {
        response.json().then(function (data) {
            let optionHTML = '';
            for (let ps of data.pert_sizes) {
                optionHTML += '<option value="' + ps[1] + '" selected=true>' + ps[0] + '</option>';
            }
            pert_size_picker.html(optionHTML);
            pert_size_picker.selectpicker('refresh');
            pert_size_picker.selectpicker('render');
        });
    });
}

function checkPert() {
    checkAttacks();
    checkPertSizes();
}

pert_type_select.onclick = checkPert;
$(document).ready(checkPert);
