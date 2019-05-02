python3 trainer.py --batch_size 16 --gpus 0   --num_threads 1 --log_mode w
python3 trainer.py --batch_size 16 --gpus 0   --num_threads 4
python3 trainer.py --batch_size 16 --gpus 0   --num_threads 16

python3 trainer.py --batch_size 16 --gpus 0,1 --num_threads 1
python3 trainer.py --batch_size 16 --gpus 0,1 --num_threads 4
python3 trainer.py --batch_size 16 --gpus 0,1 --num_threads 16

python3 trainer.py --batch_size 32 --gpus 0   --num_threads 1
python3 trainer.py --batch_size 32 --gpus 0   --num_threads 4
python3 trainer.py --batch_size 32 --gpus 0   --num_threads 16

python3 trainer.py --batch_size 32 --gpus 0,1 --num_threads 1
python3 trainer.py --batch_size 32 --gpus 0,1 --num_threads 4
python3 trainer.py --batch_size 32 --gpus 0,1 --num_threads 16

python3 trainer.py --batch_size 64 --gpus 0   --num_threads 1
python3 trainer.py --batch_size 64 --gpus 0   --num_threads 4
python3 trainer.py --batch_size 64 --gpus 0   --num_threads 16

python3 trainer.py --batch_size 64 --gpus 0,1 --num_threads 1
python3 trainer.py --batch_size 64 --gpus 0,1 --num_threads 4
python3 trainer.py --batch_size 64 --gpus 0,1 --num_threads 16

echo "SUMMARY:"
cat logs.txt | grep TIMING | tee profile.txt