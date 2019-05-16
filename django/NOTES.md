
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

6. define data model - database layout, metadata.
Don’t repeat yourself (DRY)
Every distinct concept and/or piece of data should live in one, and only one, place.
Redundancy is bad. Normalization is good.
The framework, within reason, should deduce as much as possible from as little as possible.

create two models: Question and Choice ==> polls/models.py
A Question has a question and a publication date. 
A Choice has two fields: the text of the choice and a vote tally. 
Each Choice is associated with a Question.

class inherits django.db.models.Model --> inherits model db fields 
e.g. <fieldname or db columns name> = models.CharField
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')
    
    Here, pub_date has a verbose_name so don't need python introspection
    models.DateTimeField(verbose_name=None, name=None, auto_now=False, auto_now_add=False, **kwargs)
    models.CharField(*args, **kwargs)
    
    Some Field classes have required arguments.
    CharField, for example, requires that you give it a max_length. 
    That’s used not only in the database schema, but in validation

    A Field can also have various optional arguments;
    votes = models.IntegerField(default=0)

    accessing relations: https://docs.djangoproject.com/en/2.0/ref/models/relations/ 

    relationship is defined, using ForeignKey. 
    each Choice is related to a single Question. 
    Django supports all the common database relationships: 
    many-to-one, many-to-many, and one-to-one.

    class Choice(models.Model):
        question = models.ForeignKey(Question, on_delete=models.CASCADE)

7. install app
    Django apps are “pluggable”: You can use an app in multiple projects, 
    and you can distribute apps, because they don’t have to be tied to 
    a given Django installation.

    <app name>/apps.py ==> class appConfig(django.apps.AppConfig)
    add to INSTALLED_APPS in mysite/setting.py
    python manage.py migrate ==> synchronizing models changes with the schema in the database

python manage.py makemigrations polls ==> made some changes to your models 
(base) C:\Users\yeoshuiming\Dropbox\GitHub\py-learn\django\project1\mysite>python manage.py makemigrations polls
Migrations for 'polls':
  polls\migrations\0001_initial.py
    - Create model Choice
    - Create model Question
    - Add field question to choice

Migrations are how Django stores changes to your models 
(and thus your database schema) - they’re just files on disk. 
You can read the migration for your new model if you like; 
it’s the file polls/migrations/0001_initial.py.
they’re designed to be human-editable in case you want to 
manually tweak how Django changes things.

python manage.py sqlmigrate polls 0001 ==> tailored to the db u are using, in this case, its SQLite
(base) C:\Users\yeoshuiming\Dropbox\GitHub\py-learn\django\project1\mysite>python manage.py sqlmigrate polls 0001
BEGIN;
--
-- Create model Choice
--
CREATE TABLE "polls_choice" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "choice_text" varchar(200) NOT NULL, "votes" integer NOT NULL);
--
-- Create model Question
--
CREATE TABLE "polls_question" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "question_text" varchar(200) NOT NULL, "pub_date" datetime NOT NULL);
--
-- Add field question to choice
--
ALTER TABLE "polls_choice" RENAME TO "polls_choice__old";
CREATE TABLE "polls_choice" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "choice_text" varchar(200) NOT NULL, "votes" integer NOT NULL, "question_id" integer NOT NULL REFERENCES "polls_question" ("id") DEFERRABLE INITIALLY DEFERRED);
INSERT INTO "polls_choice" ("id", "choice_text", "votes", "question_id") SELECT "id", "choice_text", "votes", NULL FROM "polls_choice__old";
DROP TABLE "polls_choice__old";
CREATE INDEX "polls_choice_question_id_c5b4b260" ON "polls_choice" ("question_id");
COMMIT;

python manage.py check;
(base) C:\Users\yeoshuiming\Dropbox\GitHub\py-learn\django\project1\mysite>python manage.py check
System check identified no issues (0 silenced).

three-step guide to making model changes:
- Change your models (in models.py).
- Run python manage.py makemigrations to create migrations for those changes
- Run python manage.py migrate to apply those changes to the database.


database API: https://docs.djangoproject.com/en/2.0/topics/db/queries

from polls.models import Question, Choice

from django.utils import timezone
q = Question(question_text="What's new?", pub_date=timezone.now())

# Save the object into the database. You have to call save() explicitly.
q.save()

# Access model field values via Python attributes.
q.question_text
q.pub_date

# Change values by changing the attributes, then calling save().
q.question_text = "What's up?"
q.save()

# show all questions
Question.objects.all()
#Out[11]: <QuerySet [<Question: Question object (1)>]>


In [3]: Question.objects.all()
#Out[3]: <QuerySet [<Question: What's new?>, <Question: What's new?>]>

Question.objects.filter(id=1)
Question.objects.get(pk=1)

q.choice_set.create(choice_text='Not much', votes=0)
q.choice_set.create(choice_text='The sky', votes=0)
c = q.choice_set.create(choice_text='Just hacking again', votes=0)
q.choice_set.all()
q.choice_set.count()

# Use double underscores to separate relationships.
# This works as many levels deep as you want; there's no limit.
Choice.objects.filter(question__pub_date__year=current_year)

# Let's delete one of the choices. Use delete() for that.
c = q.choice_set.filter(choice_text__startswith='Just hacking')
c.delete()


8. create admin user

python manage.py createsuperuser

(base) C:\Users\yeoshuiming\Dropbox\GitHub\py-learn\django\project1\mysite>python manage.py createsuperuser
Username (leave blank to use 'yeoshuiming'):
Email address: yeoshuiming@gmail.com
Password:
Password (again):
This password is too short. It must contain at least 8 characters.
This password is too common.
This password is entirely numeric.
Password:
Password (again):
Error: Your passwords didn't match.
Password:
Password (again):
Superuser created successfully.
