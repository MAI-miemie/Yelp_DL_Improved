@echo off
echo ========================================
echo 深度学习推荐系统安装脚本（大样本版本）
echo ========================================
echo.

echo 1. 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误：未找到Python，请先安装Python
    pause
    exit /b 1
)

echo.
echo 2. 激活虚拟环境...
call yelp_recommendation_env\Scripts\activate

echo.
echo 3. 升级pip...
python -m pip install --upgrade pip

echo.
echo 4. 清理之前的安装...
pip uninstall scikit-surprise implicit -y 2>nul

echo.
echo 5. 安装简化版本的依赖...
pip install -r requirements.txt

echo.
echo 6. 安装wandb（如果缺失）...
pip install wandb

echo.
echo 7. 下载大样本演示数据...
python download_yelp_data.py

echo.
echo 8. 验证安装...
python -c "import pandas, numpy, torch, transformers, networkx, community, sklearn, wandb; print('所有核心包安装成功！')"
if %errorlevel% neq 0 (
    echo 警告：部分包安装可能有问题，但继续运行...
)

echo.
echo 9. 运行大样本项目演示...
python main.py

echo.
echo ========================================
echo 安装和运行完成！
echo ========================================
echo.
echo 大样本数据文件已下载到 data/ 目录：
echo - 商户数据: data/yelp_business.csv (1500+ 商户)
echo - 评论数据: data/reviews_of_restaurants.txt (15000+ 评论)
echo - 用户数据: data/users.txt (3000+ 用户)
echo.
echo 大样本训练特点：
echo - 支持10000+样本训练
echo - 数据分片处理
echo - 内存优化管理
echo - 多进程加速
echo - 早停机制
echo.
pause
