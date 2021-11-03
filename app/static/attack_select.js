let pert_type_select = document.getElementById('pert_type')
let attack_select = document.getElementById('attack')
let attack_title = document.getElementById('attack_title')
attack_select.style.visibility = 'hidden'
attack_title.style.visibility = 'hidden'
pert_type_select.onclick = function () {
    pert_type = pert_type_select.value;
    fetch('/attacks/' + pert_type).then(function (response) {
        response.json().then(function (data) {
            let optionHTML = '';

            for (let attack of data.attacks) {
                optionHTML += '<option value="' + attack[1] + '">' + attack[0] + '</option>';
            }
            attack_select.style.visibility = 'visible'
            attack_title.style.visibility = 'visible'
            attack_select.innerHTML = optionHTML;
        });
    });
};

