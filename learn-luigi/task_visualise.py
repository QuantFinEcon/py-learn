
# python task_visualise.py --local-scheduler NameSubstituter --name JUSTIN
# python task_visualise.py --scheduler-host localhost NameSubstituter --name JUSTIN

# start daemon ==> cmd    luigid
# (base) C:\Users\yeoshuiming>luigid
# Defaulting to basic logging; consider specifying logging_conf_file in luigi.cfg.
# 2018-05-08 23:45:36,755 luigi.scheduler[332240] INFO: No prior state file exists at /var/lib/luigi-server/state.pickle. Starting with empty state
# 2018-05-08 23:45:36,786 luigi.server[332240] INFO: Scheduler starting up

# localhost:8082

# dependency graph NameSubstituter_JUSTIN_a2a6716098

import luigi
import time

class HelloWorld(luigi.Task):
    def requires(self):
        return None
    def output(self):
        return luigi.LocalTarget('helloworld.txt')
    def run(self):
        with self.output().open('w') as outfile:
            outfile.write('Hello World!!\n')

class NameSubstituter(luigi.Task):
    name = luigi.Parameter()

    def requires(self):
        return HelloWorld()
    def output(self):
        return luigi.LocalTarget(self.input().path + '.name_' + self.name)
    def run(self):
        time.sleep(50)
        with self.input().open() as infile, self.output().open('w') as outfile:
            text = infile.read()
            text = text.replace('World', self.name)
            outfile.write(text)
        time.sleep(50)

if __name__ == '__main__':
    luigi.run()
    