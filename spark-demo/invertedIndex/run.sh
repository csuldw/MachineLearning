hadoop fs -put data/input.data /home/hadoop-news/liudiwei/test
exe_cores=2
exe_num=2
tmp_dir=/home/hdp-guanggao/old_jobs/tmp
exe_mem=2G
drv_mem=3G

spark-submit \
    --master yarn-client \
    --driver-memory $drv_mem \
    --executor-memory $exe_mem \
    --num-executors $exe_num \
    --executor-cores $exe_cores \
    --driver-java-options -Dsun.io.serialization.extendedDebugInfo=true \
    --driver-java-options -Djava.io.tmpdir=$tmp_dir \
    --conf spark.eventLog.enabled=true \
    --conf spark.storage.memoryFraction=0.1 \
    --jars "deps/json4s-native_2.10-3.2.10.jar" \
    --class "InvertedIndex" \
    ./target/scala-2.10/invertedindex_2.10-1.0.0.jar \
    conf/base.conf
