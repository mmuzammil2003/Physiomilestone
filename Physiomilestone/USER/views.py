from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password
from django.contrib.auth import get_user_model,logout

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
User = get_user_model()  # get your CustomUser model
from .forms import CustomusercreationForm

def home(request):
    return render(request, "USER/home.html")

def register_view(request):
    if request.method == "POST":
        form=CustomusercreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
        else:
            print(form.errors)
    else:
        form=CustomusercreationForm()
    return render(request, "USER/register.html",{"form":form})

def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)

            # Redirect based on role
            if user.role == "child":
                return redirect("cdashboard")   # goes to Child/Dashboard
            elif user.role == "doctor":
                return redirect("ddashboard")   # goes to Doctor/Dashboard
            else:
                messages.error(request, "Role not recognized!")
                return redirect("login")
        else:
            messages.error(request, "Invalid username or password")

    return render(request, "USER/login.html")

def logoutview(request):
    logout(request)
    return redirect("login")
        