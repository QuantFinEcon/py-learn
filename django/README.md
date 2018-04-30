Django is a web framework in python, so basically it helps you to develop web applications in python. Now you must be thinking why you would use a web framework. You can write web application directly in python, to develop a web application you need the following ingredients

A WSGI (web server gateway interface) It is a standard interface between web servers and Python web applications or frameworks, to promote web application portability across a variety of web servers.

Routing: You need to route incoming HTTP requests to the right handlers, by the right handlers I mean the code which will process a particular request. for e.g. www.example.com/index where /index should to be routed to a particular python code which handles it, it can be anything like displaying a list.

SQL interface, you can use other nosql database but for most applications a sql database just serves the purpose. So you need an interface to that database

Template: To make dynamic pages you need template libraries which can populate the html page with the data that you send from your python functions (handlers)

1. The Model Layer,
2. The Views Layer,
3. The Template Layer 
4. Forms
5. The Development Process
6. The Admin
7. Security
8. Internationalization and Localization
9. Performance and Optimization
10. Python compatibility
11. Geographic framework
12. Common Web Application Tools
13. Other Core Functionalities

So Django is a collection of all these libraries and a lot of other stuff too so that you can mostly concentrate on what your application does instead of other boiler plate stuff like these. Also there are other third party libraries which you can include in your django project to make your task easier. e.g. user registration etc.

The best resource for learning django is of course The Web framework for perfectionists with deadlines is [*DjangoProject](https://docs.djangoproject.com)

Other References:
- Django Unleashed
- Two Scoops of Django Best Practices For Django 1.11
