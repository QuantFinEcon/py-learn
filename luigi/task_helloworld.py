import luigi

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
        with self.input().open() as infile, self.output().open('w') as outfile:
            text = infile.read()
            text = text.replace('World', self.name)
            outfile.write(text)

if __name__ == '__main__':
    luigi.run()
    # task = NameSubstituter()
    # task.run()

""" 
Uses argparse --kwarg <input>
python task_helloworld.py --local-scheduler HelloWorld
python task_helloworld.py --local-scheduler NameSubstituter --name JUSTIN
runs all dependencies...
naming ending with _ver01 ... helps with order of task flow
"""