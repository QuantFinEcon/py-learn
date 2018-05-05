"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

""" path(route, view, kwargs=None, name=None, *, 
Pattern=<class 'django.urls.resolvers.RoutePattern'>)

@route: path of URL, search list in order until find match
admin/ ==> ip:port/admin/<anything..>
'' ==> at its root
@view: calls the specified view function with an HttpRequest object as the first argument
@name: Naming your URL lets you refer to it unambiguously from 
    elsewhere in Django, especially from within templates
@kwargs: dict(kw, args)
"""
 
urlpatterns = [
    path(route='admin/', view=admin.site.urls),
    path(route='polls/', view=include('polls.urls'))
]
