from training import PipelineRunner as p  # type: ignore[import-not-found]

if __name__ == '__main__':
    runner = p.PipelineRunner()
    runner.start_pipline()
