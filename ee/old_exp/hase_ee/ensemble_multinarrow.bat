
chcp 65001

REM "自宅用"

rem 仮想環境をActivateするための特殊なバッチファイルを起動
rem call C:\Users\himat\anaconda3\Scripts\activate.bat

rem 仮想環境をActivate
call activate torch2

rem D: 
rem cd %0

rem python main_ensembles.py T modelname dataname div ensembles smooth
REM call python main_ensemble.py 1 resnet18 dtd 1 1 0.0 False

REM T type model data pretrained T-type label_smoothing

call python ensemble_multinarrow.py 1 resnet18 cifar100 4 dist False 5000
call python ensemble_multinarrow.py 1 resnet18 cifar100 16 lambda False 5000

rem コマンドプロンプトの画面を残す場合（残さない場合不要）
pause


@echo on

rem python main_tinyimagenet.py 1 "None" "resnet50" "dtd" True ; python main_tinyimagenet.py 45.254833995939045 "None" "resnet50" "dtd" True ; python main_tinyimagenet.py 1 "LN" "resnet50" "dtd" True ; python main_tinyimagenet.py 28.053974329660274 "BN" "resnet50" "dtd" True ; python main_tinyimagenet.py 21.967556368642413 "BN" "resnet50" "dtd" True ; python main_tinyimagenet.py 1 "None" "resnet50" "dtd" True ; python main_tinyimagenet.py 45.254833995939045 "None" "resnet50" "dtd" True ; python main_tinyimagenet.py 1 "LN" "resnet50" "dtd" True ; python main_tinyimagenet.py 28.053974329660274 "BN" "resnet50" "dtd" True ; python main_tinyimagenet.py 21.381614018462297 "BN" "resnet50" "dtd" True ; python main_tinyimagenet.py 1 "None" "resnet50" "dtd" True ; python main_tinyimagenet.py 45.254833995939045 "None" "resnet50" "dtd" True ; python main_tinyimagenet.py 1 "LN" "resnet50" "dtd" True ; python main_tinyimagenet.py 28.053974329660274 "BN" "resnet50" "dtd" True ; python main_tinyimagenet.py 18.868659892587825 "BN" "resnet50" "dtd" True ; python main_tinyimagenet.py 1 "None" "resnet50" "oxfordpet" True ; python main_tinyimagenet.py 45.254833995939045 "None" "resnet50" "oxfordpet" True ; python main_tinyimagenet.py 1 "LN" "resnet50" "oxfordpet" True ; python main_tinyimagenet.py 28.053974329660274 "BN" "resnet50" "oxfordpet" True ; python main_tinyimagenet.py 19.12397850637752 "BN" "resnet50" "oxfordpet" True ; 
rem python main_tinyimagenet.py 1 "None" "resnet50" "gtsrb" True ; python main_tinyimagenet.py 45.254833995939045 "None" "resnet50" "gtsrb" True ; python main_tinyimagenet.py 1 "LN" "resnet50" "gtsrb" True ; python main_tinyimagenet.py 28.053974329660274 "BN" "resnet50" "gtsrb" True ; python main_tinyimagenet.py 24.78562806952047 "BN" "resnet50" "gtsrb" True

