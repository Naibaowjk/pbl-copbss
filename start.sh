export DISPLAY=`cat /etc/resolv.conf | grep nameserver | awk '{print $2}'`:0
python3 client.py simulator --run_FastICA --run_MeICA --source_num 3
python3 client.py simulator --mode ALL -source_num 8 --test_num 3 --run_FastICA --run_MeICA --service_latency 150 --micro_latency 50 --service_performance 5 
python3 client.py simulator --mode ALL --run_MeICA --source_num 7 --service_latency 150 --micro_latency 20 --service_performance 4