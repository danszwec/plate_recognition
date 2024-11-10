import time
from collections import defaultdict, deque
from operator import itemgetter

class TimeManager:
    def __init__(self, rolling_avg_sessions=10):
        self.start_times = {}
        self.end_times = {}
        self.total_times = {}
        self.rolling_avg_window = rolling_avg_sessions
        self.rolling_average_data = defaultdict(lambda: deque(maxlen=self.rolling_avg_window))

    def start(self, session_name):
        try:
            self.start_times[session_name] = time.time()
        except Exception as e:
            print(f"TimeManager error in start: {e}")

    def stop(self, session_name):
        try:
            if session_name not in self.start_times:
                print(f"No start time recorded for '{session_name}'.")
                return
            end_time = time.time()
            elapsed = (end_time - self.start_times[session_name]) * 1000  # Convert to ms
            self.end_times[session_name] = end_time
            self.total_times[session_name] = elapsed
            self.rolling_average_data[session_name].append(elapsed)
        except Exception as e:
            print(f"TimeManager error in stop: {e}")
    
    def add_session(self, session_name, elapsed):
        try:
            # Check if the start_session has been started
            # if session_name in self.start_times:
            #     print(f"Warning: Session '{session_name}' already exsits.")
            #     return
          
            start_time = None
            end_time = None
            
            if isinstance(elapsed, list) and len(elapsed)==3:
                start_time = elapsed[0]
                end_time = elapsed[1]
                elapsed = elapsed[2]

            self.start_times[session_name] = start_time
            self.end_times[session_name] = end_time
            self.total_times[session_name] = elapsed
            self.rolling_average_data[session_name].append(elapsed)
        
        except Exception as e:
            print(f"add_session error: {e}")
              
    def elapsed_time(self, start_session, new_session=None):
        try:
            # Check if the start_session has been started
            if start_session not in self.start_times:
                print(f"Warning: Session '{start_session}' was never started.")
                return
            
            # Calculate elapsed time from start_session's start time
            elapsed = (time.time() - self.start_times[start_session]) * 1000
            
            # If no new session is provided, we just update the end time and total time for start_session
            if new_session is None:
                self.end_times[start_session] = time.time()
                self.total_times[start_session] = elapsed
                self.rolling_average_data[start_session].append(elapsed)
            else:
                # Otherwise, we update the total time for new_session without stopping start_session
                self.total_times[new_session] = elapsed
                self.end_times[new_session] = time.time()
                self.rolling_average_data[new_session].append(elapsed)
        except Exception as e:
            print(f"TimeManager error in elapsed_time: {e}")

    
    def get_rolling_average(self, session_name):
        try:
            data = self.rolling_average_data[session_name]
            return sum(data) / len(data) if data else None
        except Exception as e:
            print(f"TimeManager error in get_rolling_average: {e}")
            return None

    def clear(self):
        self.start_times = {}
        self.end_times = {}


    def get_total_duration(self):
        total_start = min(self.start_times.values())
        total_end = max(self.end_times.values())
        total_time = (total_end - total_start) * 1000
        return total_time


    def get_total_duration_between(self, start_session, end_session=None):
        try:
            if not start_session in self.start_times:
                return None
            if end_session:
                end_time = self.end_times.get(end_session, time.time())
            else:
                end_time = time.time()

            sessions = list(self.start_times.keys())
            idx_start = sessions.index(start_session)
            idx_end = sessions.index(end_session) if end_session else len(sessions) - 1

            total_duration = 0
            last_end_time = self.start_times[start_session]

            for i in range(idx_start, idx_end + 1):
                session_name = sessions[i]
                current_start = self.start_times[session_name]
                current_end = self.end_times.get(session_name, None)
                if not current_end:
                    continue

                # Overlap check
                if current_start < last_end_time:
                    overlap_duration = (last_end_time - current_start) * 1000
                    total_duration += self.total_times[session_name] - overlap_duration
                else:
                    total_duration += self.total_times[session_name]
                last_end_time = current_end

            return total_duration
        except Exception as e:
            print(f"TimeManager error in get_total_duration_between: {e}")
            return None

    def summary(self, frame_no, show=True):
        total_fps = 0
        try:
            if show and frame_no%10==0:
                print("\n----- TimeManager Summary -----\n")
                header = "{:<25} {:<15} {:<20}".format("Session Name", "Last Run[ms]", f"Average[ms] [last {self.rolling_avg_window} runs]")
                print(header)
                print('-' * len(header))

                # test
                sorted_start_times = sorted(self.start_times.items(), key=itemgetter(1))
                sorted_start_times = dict(sorted_start_times)

                for session, time in sorted_start_times.items():
                    avg = self.get_rolling_average(session)
                    time = self.total_times[session]
                    row = "{:<25} {:<15.2f} {:<20.2f}".format(session, time, avg)
                    print(row)
            
            total_duration = self.get_total_duration()
            total_fps = round(1000 / total_duration , 3)
            if show and frame_no%10==0:
                print("\nTotal duration time: {:.2f} ms  | FPS: {:.2f}".format(total_duration, total_fps))
                print("-----------------------------------------------------------------\n")
        except Exception as e:
            print(f"TimeManager error in summary: {e}")
            
        return total_fps
    

    def print_timeline(self):
        process_names = sorted(self.start_times.keys())
        start_time = min(self.start_times.values(), default=0)
        end_time = max(self.end_times.values(), default=0)

        num_steps = int((end_time - start_time) * 100)  # 1 unit = 0.01 seconds
        timeline = [' '] * num_steps

        for process_name in process_names:
            try:
                start_step = int((self.start_times[process_name] - start_time) * 100)
                end_step = int((self.end_times[process_name] - start_time) * 100)
                end_step = min(end_step, num_steps)  # Ensure not to exceed the timeline length
                timeline[start_step:end_step] = [process_name] * (end_step - start_step)
            except KeyError:
                pass  # Ignore missing process start or end times

        print("time axis[ms] :")
        print(' ' * 39, f"{start_time:.2f}")
        print('_' * 45)
        for step in range(0, num_steps, 10):
            time_marker = step / 100
            timeline_segment = ''.join(timeline[step:step + 10])
            if timeline_segment:
                print(f"{timeline_segment: <35} {'-' * 12} {time_marker + start_time:.2f}")


