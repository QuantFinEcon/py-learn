
1. setup - auto-generate some code that establishes a 
    Django project directory.

    django-admin startproject mysite

    - manage.py: command line utility
    - mysite/__init__.py: make dir a package
    - ../settings.py: configuration API
    - ../url.py: "table of contents"
    - ../wsgi.py: entry point for WSGI-compatible web servers to serve your project

2. run server
    python manage.py runserver <ip:port>

    Auto reloads pyscripts on each RESTful API request

    Performing system checks...

    System check identified no issues (0 silenced).

    You have 14 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions.
    Run 'python manage.py migrate' to apply them.
    May 03, 2018 - 00:40:07
    Django version 2.0.4, using settings 'mysite.settings'
    Starting development server at http://127.0.0.1:8000/
    Quit the server with CTRL-BREAK.
    [03/May/2018 00:40:50] "GET / HTTP/1.1" 200 16348
    [03/May/2018 00:40:50] "GET /static/admin/css/fonts.css HTTP/1.1" 200 400
    [03/May/2018 00:40:51] "GET /static/admin/fonts/Roboto-Light-webfont.woff HTTP/1.1" 200 81348
    [03/May/2018 00:40:51] "GET /static/admin/fonts/Roboto-Regular-webfont.woff HTTP/1.1" 200 80304
    [03/May/2018 00:40:51] "GET /static/admin/fonts/Roboto-Bold-webfont.woff HTTP/1.1" 200 82564
    Not Found: /favicon.ico
    [03/May/2018 00:40:51] "GET /favicon.ico HTTP/1.1" 404 1972

3. create app
    A project is a collection of configuration 
    and apps for a particular website. A project 
    can contain multiple apps. An app can be in 
    multiple projects.

    create <app name> dir
    python manage.py startapps <app name>

    polls/
        __init__.py
        admin.py
        apps.py
        migrations/
            __init__.py
        models.py
        tests.py
        views.py

4. create view -> map URL in URLconf or urls.py
    <app name>/views.py
    <app name>/urls.py
    site/urls.py --> URLconf point to app/urls.py with include()
    http://localhost:8000/admin/ ==> urlpatterns = [path('admin/', admin.site.urls)]
    http://localhost:8000/<app name>/ ==> urlpatterns = [path('<app name>/', include(<lib.module>)]

5. setup db via site/settings.py
    https://docs.djangoproject.com/en/2.0/topics/install/#database-installation
    prefer more scalable database like PostgreSQL, to avoid database-switching headaches down the road.
    - BASE_DIR
    - DATABASES
    - WSGI_application
    - ROOT_URLCONF
    - DEBUG
    - INSTALLED_APPS ==> like  __init__ auth, sessions, messages, staticfiles, admin, contenttypes
    - TEMPLATES
    - internationalisation

python manage.py migrate
The migrate command looks at the INSTALLED_APPS setting and creates 
any necessary database tables according to the database settings in 
your mysite/settings.py file and the database migrations shipped with the app

(base) C:\Users\yeoshuiming\Dropbox\GitHub\py-learn\django\project1\mysite>python manage.py migrate
Operations to perform:
  Apply all migrations: admin, auth, contenttypes, sessions
Running migrations:
  Applying contenttypes.0001_initial... OK
  Applying auth.0001_initial... OK
  Applying admin.0001_initial... OK
  Applying admin.0002_logentry_remove_auto_add... OK
  Applying contenttypes.0002_remove_content_type_name... OK
  Applying auth.0002_alter_permission_name_max_length... OK
  Applying auth.0003_alter_user_email_max_length... OK
  Applying auth.0004_alter_user_username_opts... OK
  Applying auth.0005_alter_user_last_login_null... OK
  Applying auth.0006_require_contenttypes_0002... OK
  Applying auth.0007_alter_validators_add_error_messages... OK
  Applying auth.0008_alter_user_username_max_length... OK
  Applying auth.0009_alter_user_last_name_max_length... OK
  Applying sessions.0001_initial... OK

