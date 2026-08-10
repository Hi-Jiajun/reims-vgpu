/*
 * hover.c — glide the pointer onto a point and rest it there, so the guest
 * raises a real hover effect.
 *
 * A dock hover is not a mouse position, it is an *arrival followed by rest*.
 * The window server starts its tooltip timer when the pointer stops moving over
 * a target, so setting the cursor once and returning measures nothing: the
 * effect this probe exists to photograph has not been asked for yet. Equally, a
 * program that keeps re-posting the same coordinate every frame restarts that
 * timer forever and the tooltip never appears. So this glides in over a short
 * approach and then genuinely stops posting.
 *
 * C rather than Python for `drag.c`'s reason: the guest's /usr/bin/python3 has
 * no PyObjC, so `import Quartz` fails, while clang and the ApplicationServices
 * headers are present. Compiled on the guest by the harness; nothing is shipped
 * or committed as a binary.
 *
 * Prints one line of JSON, so a run that posted nothing cannot be mistaken for a
 * run whose hover simply had no effect.
 *
 * Build:
 *   clang -O2 -o hover hover.c -framework ApplicationServices -lm
 */
#include <ApplicationServices/ApplicationServices.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static void move_to(double x, double y)
{
    CGEventRef e = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved,
                                           CGPointMake(x, y),
                                           kCGMouseButtonLeft);
    if (!e) {
        return;
    }
    CGEventPost(kCGHIDEventTap, e);
    CFRelease(e);
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: hover X Y APPROACH_FROM_Y REST_SECONDS\n");
        return 2;
    }
    const double x = atof(argv[1]), y = atof(argv[2]);
    const double from_y = atof(argv[3]), rest = atof(argv[4]);
    if (rest <= 0.0) {
        fprintf(stderr, "hover: rest seconds must be positive\n");
        return 2;
    }

    /* The approach. Twenty steps over ~200 ms is enough for the window server to
     * see a pointer entering the target rather than teleporting into it, which
     * some hover effects require and all of them are entitled to. */
    const int steps = 20;
    const double start = now_s();
    for (int i = 0; i <= steps; i++) {
        const double t = (double)i / (double)steps;
        move_to(x, from_y + (y - from_y) * t);
        usleep(10000);
    }

    /* The rest. Deliberately no posting: this is the interval in which the
     * tooltip timer runs and the hover highlight is composited, and a re-post
     * here would restart it. */
    usleep((useconds_t)(rest * 1e6));

    printf("{\"x\":%.1f,\"y\":%.1f,\"approach_steps\":%d,\"rest_s\":%.2f,"
           "\"elapsed\":%.3f}\n",
           x, y, steps, rest, now_s() - start);
    return 0;
}
