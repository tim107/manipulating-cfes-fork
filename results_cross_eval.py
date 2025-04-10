python evaluate_models.py --dataset "cc" --model_path "models/cc_wachter" --cf_algo "wachter" 
Protected tensor(21.4765) tensor(785.9680)
Not-Protected: tensor(21.9357) Var: tensor(388.4911)
Not-Protected + Delta: tensor(7.6823) Var: tensor(6.8212e-13)
Delta tensor(0.4592)
Cost Ratio tensor(2.8553)
python evaluate_models.py --dataset "cc" --model_path "models/cc_wachter" --cf_algo "wachter-sparse"
Protected tensor(567432.2500) tensor(2.1627e+13)
Not-Protected: tensor(1864.8497) Var: tensor(68067040.)
Not-Protected + Delta: tensor(10.6784) Var: tensor(1.3318e-12)
Delta tensor(565567.3750)
Cost Ratio tensor(174.6375)
python evaluate_models.py --dataset "cc" --model_path "models/cc_wachter" --cf_algo "proto"
Protected tensor(6.3152) tensor(2.6108)
Not-Protected: tensor(4.6222) Var: tensor(2.9238)
Not-Protected + Delta: tensor(14.7824) Var: tensor(1.9033)
Delta tensor(1.6930)
Cost Ratio tensor(0.3127)
python evaluate_models.py --dataset "cc" --model_path "models/cc_wachter" --cf_algo "revise"
Protected tensor(5.9901) tensor(nan)
Not-Protected: tensor(nan) Var: tensor(nan)
Not-Protected + Delta: tensor(0.6263) Var: tensor(nan)
Delta tensor(nan)
Cost Ratio tensor(nan)
#################################################################
python evaluate_models.py --dataset "cc" --model_path "models/cc_wachter-sparse" --cf_algo "wachter"
Protected tensor(21.6279) tensor(1791.1025)
Not-Protected: tensor(128.3511) Var: tensor(295088.0938)
Not-Protected + Delta: tensor(2.5294) Var: tensor(1.7220e-13)
Delta tensor(106.7232)
Cost Ratio tensor(50.7442)
python evaluate_models.py --dataset "cc" --model_path "models/cc_wachter-sparse" --cf_algo "wachter-sparse"
Protected tensor(1.1322e+10) tensor(6.1990e+21)
Not-Protected: tensor(1.4657e+13) Var: tensor(7.5187e+27)
Not-Protected + Delta: tensor(3.3957) Var: tensor(1.5381e-13)
Delta tensor(1.4645e+13)
Cost Ratio tensor(4.3163e+12)
python evaluate_models.py --dataset "cc" --model_path "models/cc_wachter-sparse" --cf_algo "proto"
Protected tensor(6.3152) tensor(28.0115)
Not-Protected: tensor(10.1419) Var: tensor(760.1484)
Not-Protected + Delta: tensor(7.6053) Var: tensor(0.8570)
Delta tensor(3.8267)
Cost Ratio tensor(1.3335)
python evaluate_models.py --dataset "cc" --model_path "models/cc_wachter-sparse" --cf_algo "revise"
Protected tensor(5.8460) tensor(nan)
Not-Protected: tensor(0.7763) Var: tensor(nan)
Not-Protected + Delta: tensor(0.5753) Var: tensor(nan)
Delta tensor(5.0698)
Cost Ratio tensor(1.3492)
##################################################################
python evaluate_models.py --dataset "cc" --model_path "models/cc_proto" --cf_algo "wachter"
Protected tensor(277.2705) tensor(1664735.6250)
Not-Protected: tensor(3332.7512) Var: tensor(2.4973e+08)
Not-Protected + Delta: tensor(1.8141) Var: tensor(7.1623e-14)
Delta tensor(3055.4807)
Cost Ratio tensor(1837.1626)
python evaluate_models.py --dataset "cc" --model_path "models/cc_proto" --cf_algo "wachter-sparse"
Protected tensor(6076349.5000) tensor(1.6974e+15)
Not-Protected: tensor(7301142.) Var: tensor(1.3858e+15)
Not-Protected + Delta: tensor(2.3896) Var: tensor(1.2051e-13)
Delta tensor(1224792.5000)
Cost Ratio tensor(3055387.)
python evaluate_models.py --dataset "cc" --model_path "models/cc_proto" --cf_algo "proto"
Protected tensor(6.9684) tensor(37.8953)
Not-Protected: tensor(7.3242) Var: tensor(142.4412)
Not-Protected + Delta: tensor(6.8033) Var: tensor(1.5201)
Delta tensor(0.3557)
Cost Ratio tensor(1.0766)
python evaluate_models.py --dataset "cc" --model_path "models/cc_proto" --cf_algo "revise"
Protected tensor(5.9762) tensor(nan)
Not-Protected: tensor(0.3757) Var: tensor(nan)
Not-Protected + Delta: tensor(0.5897) Var: tensor(nan)
Delta tensor(5.6005)
Cost Ratio tensor(0.6371)
###################################################################

python evaluate_models.py --dataset "cc" --model_path "models/cc_revise" --cf_algo "wachter"
Protected tensor(18.2744) tensor(36.4378)
Not-Protected: tensor(15.9479) Var: tensor(17.8872)
Not-Protected + Delta: tensor(40.1092) Var: tensor(7.8396)
Delta tensor(2.3265)
Cost Ratio tensor(0.3976)
python evaluate_models.py --dataset "cc" --model_path "models/cc_revise" --cf_algo "wachter-sparse"
Protected tensor(22.3633) tensor(396.5273)
Not-Protected: tensor(16.1251) Var: tensor(126.3167)
Not-Protected + Delta: tensor(54.0652) Var: tensor(1.8370)
Delta tensor(6.2382)
Cost Ratio tensor(0.2983)
python evaluate_models.py --dataset "cc" --model_path "models/cc_revise" --cf_algo "proto"
Protected tensor(17.5886) tensor(181.8365)
Not-Protected: tensor(12.3627) Var: tensor(46.2639)
Not-Protected + Delta: tensor(58.0637) Var: tensor(1.4774)
Delta tensor(5.2260)
Cost Ratio tensor(0.2129)
python evaluate_models.py --dataset "cc" --model_path "models/cc_revise" --cf_algo "revise"
Protected tensor(6.3089) tensor(nan)
Not-Protected: tensor(0.7968) Var: tensor(nan)
Not-Protected + Delta: tensor(0.7920) Var: tensor(nan)
Delta tensor(5.5121)
Cost Ratio tensor(1.0061)
##########################################################
##########################################################
python evaluate_models.py --dataset "german" --model_path "models/german_wachter" --cf_algo "wachter"
python evaluate_models.py --dataset "german" --model_path "models/german_wachter" --cf_algo "wachter-sparse"
python evaluate_models.py --dataset "german" --model_path "models/german_wachter" --cf_algo "proto"
python evaluate_models.py --dataset "german" --model_path "models/german_wachter" --cf_algo "revise"

python evaluate_models.py --dataset "german" --model_path "models/german_wachter-sparse" --cf_algo "wachter"
python evaluate_models.py --dataset "german" --model_path "models/german_wachter-sparse" --cf_algo "wachter-sparse"
python evaluate_models.py --dataset "german" --model_path "models/german_wachter-sparse" --cf_algo "proto"
python evaluate_models.py --dataset "german" --model_path "models/german_wachter-sparse" --cf_algo "revise"

python evaluate_models.py --dataset "german" --model_path "models/german_proto" --cf_algo "wachter"
python evaluate_models.py --dataset "german" --model_path "models/german_proto" --cf_algo "wachter-sparse"
python evaluate_models.py --dataset "german" --model_path "models/german_proto" --cf_algo "proto"
python evaluate_models.py --dataset "german" --model_path "models/german_proto" --cf_algo "revise"

python evaluate_models.py --dataset "german" --model_path "models/german_revise" --cf_algo "wachter"
python evaluate_models.py --dataset "german" --model_path "models/german_revise" --cf_algo "wachter-sparse"
python evaluate_models.py --dataset "german" --model_path "models/german_revise" --cf_algo "proto"
python evaluate_models.py --dataset "german" --model_path "models/german_revise" --cf_algo "revise"
