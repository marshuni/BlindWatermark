from django.shortcuts import render

import os

from django_app.settings import BASE_DIR

import platform

os_name = platform.system()


def index(request):
    # 如果没有传入方程式就呈现起始页
    if request.GET.get('chemicalEquationSrc') not in {None, ""}:
        

        # 输出
        test = "！！！！！"
        
        
        return render(request, 'index.html', {
            'test': test,
            
        })  # 返回一个html页面
    else:
        return render(request, 'start.html')  # 返回一个html页面


def about(request):
    return render(request, 'about.html', {})
