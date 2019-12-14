result_dir='../result/PageRank_WordCount'
tem_test_dir='../test_dir_PageRank_WordCount'
tem_div_dir='../div_result_PageRank_WordCount'
tem_model_dir='../model_PageRank_WordCount'
if [ ! -d $tem_test_dir ];then
mkdir $tem_test_dir
fi
if [ ! -d $tem_div_dir ];then
mkdir $tem_div_dir
fi
if [ ! -d $tem_model_dir ];then
mkdir $tem_model_dir
fi
for time in $(seq 1 1);do
for ((i=100;i<=100;i+=10));do 
if [ ! -d $tem_test_dir/$time ];then
mkdir $tem_test_dir/$time
fi
if [ ! -d $tem_div_dir/$time ];then
mkdir $tem_div_dir/$time
fi
if [ ! -d $tem_model_dir/$time ];then
mkdir $tem_model_dir/$time
fi
div_dir=$tem_div_dir/$time/$i
test_dir=$tem_test_dir/$time/$i
model_dir=$tem_model_dir/$time/$i
if [ ! -d $div_dir ];then
mkdir $div_dir
fi
if [ ! -d $test_dir ];then
mkdir $test_dir
fi


# python main.py --train_count $i --test_dir $test_dir --result_save $result_dir/$time --model_dir $model_dir
# wait
mkfifo tm1
exec 5<> tm1
rm -f tm1
for ((t=1;t<=10;t++));do
echo >&5
done
for predict_file in $(seq 0 99);do 
# echo $test_dir/${predict_file}test_x.npy
read -u5
(
python predict.py --final_result_save $result_dir/$time --test_x_dir $test_dir/${predict_file}test_x.npy --test_y_dir $test_dir/${predict_file}test_y.npy --div_result $div_dir --mode 0 --model_dir $model_dir
echo >&5
)&
done
wait
exec 5>&-
exec 5<&-
echo "count: "$i>>$result_dir/${time}_result
python predict.py --mode 1 --test_y_dir $test_dir --div_result $div_dir --prc_result $result_dir/${time}_prc --final_result_save $result_dir/${time}_result
wait
done
done
