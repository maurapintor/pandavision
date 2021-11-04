let add_preprocessing = document.getElementById('addpreprocessing')
let add_preprocessing_custom = document.getElementById('addpreprocessing-2')
let prepr_data = document.getElementById('preprInfo')
prepr_data.style.display = 'none';

function checkPrepr() {
    if (add_preprocessing_custom.checked) {
        prepr_data.style.display = '';

    } else {
        prepr_data.style.display = 'none';
    }
}

add_preprocessing.onchange = checkPrepr;

$(document).ready(checkPrepr);