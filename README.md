# Course-of-Parallel-and-Distributed
这是存放研一的课程-并行与分布式-课程用到的代码

直接右键运行，只能创建一个线程
mpiexec -n 4 python test_mpi.py  可以创建多个线程  4个

大作业：
appV3.py 前端的ui代码
mpi_blurV3.py 后端并行处理的代码

大作业：运行方式： 
python appV3.py


性能分析：
固定卷积核 ksize=21，sigma=5 ，不同进程数性能对比
p = 1 total_time_s=8.629952 comm_time_s(max)=0.000031 comp_time_s(max)=8.619437
p = 2 total_time_s=6.280245 comm_time_s(max)=0.000087 comp_time_s(max)=6.274027
p = 4 total_time_s=4.925080 comm_time_s(max)=0.002282 comp_time_s(max)=4.916890
p = 6 total_time_s=4.475618 comm_time_s(max)=0.002828 comp_time_s(max)=4.467482
p = 8 total_time_s=4.284471 comm_time_s(max)=0.003451 comp_time_s(max)=4.276757









