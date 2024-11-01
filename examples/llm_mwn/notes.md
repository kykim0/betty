
### Commands
CUDA_VISIBLE_DEVICES=0 python3 examples/llm_mwn/main.py --report_to=none --strategy=default --model_name_or_path=EleutherAI/pythia-1.4b --weight_model_name_or_path=EleutherAI/pythia-70m --max_steps=1000 --eval_steps=100 -logging_steps=50 --gradient_accumulation_steps=16 --meta_gradient_accumulation_steps=16 --output_dir=/data/kyuyoung_kim/dev/betty/examples/llm_mwn/output



CUDA_VISIBLE_DEVICES=0 python3 examples/llm_mwn/weight_eval.py --report_to=none --model_name_or_path=EleutherAI/pythia-1.4b --weight_model_name_or_path=EleutherAI/pythia-70m --output_dir=/data/kyuyoung_kim/dev/betty/examples/llm_mwn/output
