let pert_type_select = document.getElementById('pert_type')
let attack_select = document.getElementById('attack')

function checkPert() {
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

pert_type_select.onclick = checkPert;
$(document).ready(checkPert);
