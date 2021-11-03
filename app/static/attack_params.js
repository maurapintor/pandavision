        let pgd_form = document.getElementById('pgd_form')
        let cw_form = document.getElementById('cw_form')
        let attack_params_title = document.getElementById('attack_params_title')
        pgd_form.style.display = 'none';
        cw_form.style.display = 'none';
        attack_params_title.style.display = 'none';

        function checkPGD() {
            attack_type = attack_select.value;
            if (attack_type === 'pgd-linf' || attack_type === 'pgd-l2') {
                pgd_form.style.display = '';
            } else {
                pgd_form.style.display = 'none';
                attack_params_title.style.display = 'none';

            }
        }

        function checkCW() {
            attack_type = attack_select.value;
            if (attack_type === 'cw') {
                cw_form.style.display = '';
            } else {
                cw_form.style.display = 'none';
                attack_params_title.style.display = 'none';
            }
        }

        function checkAttack() {
            checkCW();
            checkPGD();
        }

        attack_select.onclick = checkAttack;
        $(document).ready(checkAttack);
