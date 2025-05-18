python step4.1.1_assemble_4option.py --output ./data/step4/Dataset_style_1_4options --input ./data/step3/3.2_valid_style_1.json --root ./data/step3/3.1_all_scripts_style_1
python step4.1.1_assemble_4option.py --output ./data/step4/Dataset_style_2_4options --input ./data/step3/3.2_valid_style_2.json --root ./data/step3/3.1_all_scripts_style_2
python step4.1.1_assemble_4option.py --output ./data/step4/Dataset_style_3_4options --input ./data/step3/3.2_valid_style_3.json --root ./data/step3/3.1_all_scripts_style_3

python step4.1.2_assemble_4option_shuffle.py --output ./data/step4/Dataset_style_1_4options_shuffle --input ./data/step3/3.2_valid_style_1.json --root ./data/step3/3.1_all_scripts_style_1
python step4.1.2_assemble_4option_shuffle.py --output ./data/step4/Dataset_style_2_4options_shuffle --input ./data/step3/3.2_valid_style_2.json --root ./data/step3/3.1_all_scripts_style_2
python step4.1.2_assemble_4option_shuffle.py --output ./data/step4/Dataset_style_3_4options_shuffle --input ./data/step3/3.2_valid_style_3.json --root ./data/step3/3.1_all_scripts_style_3

python step4.1.3_assemble_10option.py --output ./data/step4/Dataset_style_1_10options --valid_ids ./data/step3/3.2_valid_style_1.json --root ./data/step3/3.1_all_scripts_style_1
python step4.1.3_assemble_10option.py --output ./data/step4/Dataset_style_2_10options --valid_ids ./data/step3/3.2_valid_style_2.json --root ./data/step3/3.1_all_scripts_style_2
python step4.1.3_assemble_10option.py --output ./data/step4/Dataset_style_3_10options --valid_ids ./data/step3/3.2_valid_style_3.json --root ./data/step3/3.1_all_scripts_style_3