import java.util.concurrent.*;
import java.util.concurrent.locks.*;

/** One wait of SECONDS of universe time through a Java API: java UWait API SECONDS. Real time is read from
 * /proc/uptime-free CLOCK_MONOTONIC_RAW via a tiny JNI-free trick: the parent driver times the process instead, so this
 * prints the universe's elapsed time (System.nanoTime, faked) and currentTimeMillis. */
public class UWait {
    public static void main(String[] a) throws Exception {
        String api = a[0];
        long ms = (long) (Double.parseDouble(a[1]) * 1000);
        Object o = new Object();
        long n0 = System.nanoTime(), w0 = System.currentTimeMillis();
        System.out.println("READY");
        System.out.flush();
        switch (api) {
            case "java_thread_sleep": Thread.sleep(ms); break;
            case "java_object_wait": synchronized (o) { o.wait(ms); } break;
            case "java_park_nanos": LockSupport.parkNanos(ms * 1_000_000L); break;
            case "java_queue_poll": new ArrayBlockingQueue<Integer>(1).poll(ms, TimeUnit.MILLISECONDS); break;
            case "java_scheduled_executor": {
                ScheduledExecutorService s = Executors.newSingleThreadScheduledExecutor();
                s.schedule(() -> 0, ms, TimeUnit.MILLISECONDS).get();
                s.shutdown();
                break;
            }
            case "java_condition_await": {
                ReentrantLock l = new ReentrantLock();
                Condition c = l.newCondition();
                l.lock();
                try { c.await(ms, TimeUnit.MILLISECONDS); } finally { l.unlock(); }
                break;
            }
            case "java_park_until": {
                long until = System.currentTimeMillis() + ms;
                while (System.currentTimeMillis() < until) LockSupport.parkUntil(until);
                break;
            }
            case "java_await_until": {
                ReentrantLock l = new ReentrantLock();
                Condition c = l.newCondition();
                java.util.Date until = new java.util.Date(System.currentTimeMillis() + ms);
                l.lock();
                try { while (c.awaitUntil(until)) { } } finally { l.unlock(); }
                break;
            }
            default: throw new IllegalArgumentException(api);
        }
        System.out.printf("DONE %.6f %.6f%n", (System.nanoTime() - n0) / 1e9, (System.currentTimeMillis() - w0) / 1e3);
    }
}
