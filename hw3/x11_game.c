/*
gcc -Aa -I/usr/X11R6/include -I/usr/X11R6/include/X11 -L/usr/X11R6/lib  -L/usr/X11R6/lib/X11 -lX11 x11_game.c -o game -lX11 -lm
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>

/* ### Constant Define ### */

#define W0_WIDTH    500
#define W0_HEIGHT   300
#define INTERVAL    8000*4
#define RANGE       58
#define VEROCITY1   1
#define VEROCITY2   2
#define VEROCITY3   3
#define VEROCITY4   4
#define VEROCITY5   12
#define POINT1_X    W0_WIDTH*1/5
#define POINT1_Y    W0_HEIGHT*4/5
#define POINT2_X    W0_WIDTH*4/5
#define POINT2_Y    W0_HEIGHT*4/5

#define M_MAX       20
#define NO_USE      0
#define MISSILE     1
#define EXPLOSION   2

#define TITLE1      "MISSILE"
#define FONT1       "fg-16"
#define FONT2  "-adobe-courier-medium-o-normal--34-240-100-100-m-200-iso8859-1"
#define CURSOR1     XC_tcross
#define SCORE_FILE  "~/.mis_score"

/* ### Global Variables ### */

/* for DISPLAY */
Display *dpy;
Window  w0;
Pixmap  pix0, pix1;
GC      gc_b, gc_w;

/* for GAME */
struct work {
  float  x, y;
  float  vx, vy;
  float  dx, dy;
  int    flag, state;
};
struct work     missile[M_MAX*2][2], expl[M_MAX*2];

int     mis_count, mis_left, mis_right;
int     damage, level, score, hi_score;
int     timer, interval;
char    user_name[32], hi_user_name[32];

/* ### Local Function ### */

void    WindowInit();
void    PixInit();
void    Redraw(int p);
void    Destroy();
void    NullFunction();
void    Vsync();
void    SyncSet(int intr);
void    GameInit(int mode);
void    M_Set(int m, float v, float sx, float sy, float dx, float dy);
void    M_Exp(float sx, float sy);
void    M_Draw(int mode);
void    M_Disp(int mode);
void    Score_Load();
void    Score_Save();

struct  itimerval         Value;
struct  itimerval        OValue;
long    IntervalTime = INTERVAL;

/* =-=-=-=-=-=-=-=-= */
main(int argc, char **argv)
{
  XEvent  event;
  int     event_num, flag;

                                                        /* system initialize */
  if (argc >= 2) {
    sprintf(user_name, "%s", argv[1]);
  } else {
    sprintf(user_name, "%s", getenv("USER"));
  }
  fprintf(stderr, "Entry-name is [%s].\n", user_name);

  WindowInit();
  PixInit();
  GameInit(True);
  Redraw(0);

                                                                /* main loop */
  while (True) {
                                                          /* game initialize */
    GameInit(False);
    Score_Load();
                                                                /* demo loop */
    flag = True;
    while (flag) {

                                                                     /* wait */
      Vsync();

                                                 /* demo explosion & missile */
      if (timer % 3 == 0) {
    M_Exp((float)(rand()%W0_WIDTH), (float)(rand()%POINT1_Y));
      }
      if (timer % 7 == 0) {
    M_Set(0, VEROCITY4, (float)(rand()%W0_WIDTH), (float)(rand()%POINT1_Y),
          (float)(rand()%W0_WIDTH), (float)(rand()%POINT1_Y));
      }
      if ((timer % 500 == 300)||
      (timer % 500 == 305)||
      (timer % 500 == 310)||
      (timer % 500 == 315)) {
    M_Set(0, VEROCITY5, POINT1_X, POINT1_Y,
          (float)(rand()%W0_WIDTH), (float)(rand()%200+25));
      }
      if ((timer % 373 == 220)||
      (timer % 373 == 224)||
      (timer % 373 == 228)||
      (timer % 373 == 240)||
      (timer % 373 == 248)) {
    M_Set(0, VEROCITY5, POINT2_X, POINT2_Y,
          (float)(rand()%W0_WIDTH), (float)(rand()%200+25));
      }

                                                                 /* map draw */
      M_Draw(False);
      M_Disp(False);

                                                       /* draw guide message */
      if ((0 <= timer % 100)&&(timer % 100 < 70)) {
    XDrawString(dpy, pix0, gc_w, 58, 210, "PUSH MOUSE 1 BUTTON", 19);
    XDrawString(dpy, pix0, gc_w, 60, 211, "PUSH MOUSE 1 BUTTON", 19);
      }

      Redraw(0);
      XFlush(dpy);

                                                         /* event flush loop */
      event_num = XPending(dpy);
      while (event_num-- != 0) {
    XNextEvent(dpy, &event);
    if (event.type == ButtonPress) {

                                               /* ButtonPress1 -> game start */
      if (event.xbutton.button == Button1) {
        flag = False;

                                               /* ButtonPress3 -> game abort */
      } else if (event.xbutton.button == Button3) {
        Destroy();
        exit(0);
      }
    }
      }
    }

                                                          /* game initialize */
    GameInit(True);
                                                                /* game loop */
    flag = True;
    while (flag) {

                                                                     /* wait */
      Vsync();

                                                         /* event flush loop */
      event_num = XPending(dpy);
      while (event_num-- != 0) {
    XNextEvent(dpy, &event);
    if (event.type == ButtonPress) {

                                           /* ButtonPress1 -> missile set(l) */
      if ((event.xbutton.button == Button1)&&(mis_left == True)) {
        M_Set(0, VEROCITY5, POINT1_X, POINT1_Y,
          (float)event.xbutton.x,
          event.xbutton.y >= POINT1_Y-RANGE/4 ?
          (float)POINT1_Y-RANGE/4 : (float)event.xbutton.y);
        mis_count++;

                                           /* ButtonPress2 -> missile set(r) */
      } else if ((event.xbutton.button == Button2)&&(mis_right == True)) {
        M_Set(0, VEROCITY5, POINT2_X, POINT2_Y,
          (float)event.xbutton.x,
          event.xbutton.y >= POINT1_Y-RANGE/4 ?
          (float)POINT1_Y-RANGE/4 : (float)event.xbutton.y);
        mis_count++;

                                           /* ButtonPress3 -> game abort */
      } else if (event.xbutton.button == Button3) {
        flag = False;
      }
        }
      }

                                                        /* enemy missile set */
      if (timer % 48 == 0) {
    M_Set(1, VEROCITY1, (float)(rand()%(W0_WIDTH-20)+10), 0,
              (float)(rand()%(W0_WIDTH-20)+10), POINT1_Y);
      }
      if ((timer > 1000)&&(timer % 123 == 0)) {
    M_Set(1, VEROCITY2, (float)(rand()%(W0_WIDTH-20)+10), 0,
              (float)(rand()%(W0_WIDTH-20)+10), POINT1_Y);
      }
      if ((timer > 2000)&&(timer % 141 == 0)) {
    M_Set(1, VEROCITY3, (float)(rand()%(W0_WIDTH-20)+10), 0,
              (float)(rand()%(W0_WIDTH-20)+10), POINT1_Y);
      }
      if ((timer > 3000)&&(timer % 171 == 0)) {
    M_Set(1, VEROCITY4, (float)(rand()%(W0_WIDTH-20)+10), 0,
              (float)(rand()%(W0_WIDTH-20)+10), POINT1_Y);
      }

                                                           /* level up check */
      if ((mis_count == 10)&&(interval >= IntervalTime / 2)) {
    mis_count = 0;
    level += 1;
    SyncSet(interval -= 1000);
      }

                                                                 /* map draw */
      M_Draw(True);
      M_Disp(True);
      Redraw(0);
      XFlush(dpy);

                                                          /* game over check */
      if (damage >= 100*4) {
    SyncSet(interval = IntervalTime);
    timer = 0;

                                                           /* game over loop */
    flag = True;
    while (flag) {

                                                                     /* wait */
      Vsync();

                                                         /* event flush loop */
      event_num = XPending(dpy);
      while (event_num-- != 0) {
        XNextEvent(dpy, &event);

                                                      /* buttonpress -> demo */
        if ((timer > 100)&&(event.type == ButtonPress)) {
          flag = False;
        }
      }

                                                      /* game over explosion */
      if (timer % 3 == 0) {
        M_Exp((float)(rand()%W0_WIDTH), (float)(rand()%POINT1_Y));
      }

                                                                 /* map draw */
      M_Draw(False);
      M_Disp(True);

                                                       /* draw guide message */
      XDrawString(dpy, pix0, gc_w, 58, 110, " G A M E   O V E R ", 19);
      XDrawString(dpy, pix0, gc_w, 60, 111, " G A M E   O V E R ", 19);
      if (timer <= 100) {
        XDrawString(dpy, pix0, gc_w, 58, 310-timer*2,
            " G A M E   O V E R ", 19);
        XDrawString(dpy, pix0, gc_w, 60, 311-timer*2,
            " G A M E   O V E R ", 19);
      }
      if (hi_score < score) {
        XDrawString(dpy, pix0, gc_w, 50, 210-abs(6*sin(timer/3.0)),
            "you got the hi-score!", 21);
      }

      Redraw(0);
      XFlush(dpy);
    }
      }
    }
    Score_Load();
    if (hi_score < score) {
      hi_score = score;
      sprintf(hi_user_name, "%s", user_name);
      Score_Save();
    }
  }
}


/**/


void WindowInit()
{
  XGCValues       gv;
  Font    font;
  XSizeHints      hint;
  Cursor  cursor;

                                                        /* display open loop */
  if ((dpy = XOpenDisplay(NULL) ) == NULL) {
    fprintf(stderr, "Can't open display !\n");
    exit(-1);
  }

                                                              /* window open */
  w0 = XCreateSimpleWindow(dpy, RootWindow(dpy, 0),
               0, 0, W0_WIDTH, W0_HEIGHT,
               1, BlackPixel(dpy, 0), WhitePixel(dpy, 0));

                                                              /* pixmap open */
  pix0 = XCreatePixmap(dpy, w0,
               W0_WIDTH, W0_HEIGHT,
               DefaultDepth(dpy, 0));
  pix1 = XCreatePixmap(dpy, w0,
               W0_WIDTH, W0_HEIGHT,
               DefaultDepth(dpy, 0));

                                                             /* select input */
  XSelectInput(dpy, w0, ButtonPressMask);

                                                                /* create gc */
  gc_b = XCreateGC(dpy, w0, 0, NULL);
  gc_w = XCreateGC(dpy, w0, 0, NULL);
  XSetForeground(dpy, gc_b, BlackPixel(dpy, 0));
  XSetForeground(dpy, gc_w, WhitePixel(dpy, 0));
  gv.graphics_exposures  = False;
  XChangeGC(dpy, gc_b, GCForeground, &gv);
  XChangeGC(dpy, gc_w, GCBackground, &gv);

                                                                 /* set hint */
  hint.flags = PMaxSize | PMinSize;
  hint.min_width  = hint.max_width  = W0_WIDTH;
  hint.min_height = hint.max_height = W0_HEIGHT;
  XSetStandardProperties(dpy, w0, TITLE1, TITLE1, None, NULL, 0, &hint);

                                                                 /* set font */
  font = XLoadFont(dpy, FONT1);
  XSetFont(dpy, gc_b, font);
  font = XLoadFont(dpy, FONT2);
  XSetFont(dpy, gc_w, font);

                                                               /* window map */
  XMapWindow(dpy, w0);

                                                            /* change cursor */
  cursor = XCreateFontCursor(dpy, CURSOR1);
  XDefineCursor(dpy, w0, cursor);

} /* end of WindowInit */


void PixInit()
{
  XFillRectangle(dpy, pix0, gc_b, 0, 0, W0_WIDTH, W0_HEIGHT);

  XFillRectangle(dpy, pix1, gc_b, 0, 0, W0_WIDTH, W0_HEIGHT);
  XFillRectangle(dpy, pix1, gc_w, 0, POINT1_Y, W0_WIDTH, W0_HEIGHT-POINT1_Y);
  XFillArc(dpy, pix1, gc_b, POINT1_X-10, POINT1_Y, 20, 10, 0*64, 360*64);
  XFillArc(dpy, pix1, gc_b, POINT2_X-10, POINT2_Y, 20, 10, 0*64, 360*64);
  XFillRectangle(dpy, pix1, gc_w, POINT1_X-2, POINT1_Y-4, 5, 9);
  XFillRectangle(dpy, pix1, gc_w, POINT2_X-2, POINT2_Y-4, 5, 9);

} /* end of PixInit */


void Redraw(int p)
{
  if (p == 0) {
    XCopyArea(dpy, pix0, w0, gc_b, 0, 0, W0_WIDTH, W0_HEIGHT, 0, 0);
  } else {
    XCopyArea(dpy, pix1, w0, gc_b, 0, 0, W0_WIDTH, W0_HEIGHT, 0, 0);
  }

} /* end of Redraw */


void Destroy()
{
  XFreeGC(dpy, gc_b);
  XFreeGC(dpy, gc_w);
  XUndefineCursor(dpy, w0);
  XDestroyWindow(dpy, w0);
  XCloseDisplay(dpy);

} /* end of Destroy */


void NullFunction()
{
  signal(SIGALRM, NullFunction);

} /* end of NullFunction */


void Vsync()
{
  timer++;

  pause();

} /* end of Vsync */


void SyncSet(int intr)
{
  timerclear(&Value.it_interval);
  timerclear(&Value.it_value);
  Value.it_interval.tv_sec = 0;
  Value.it_interval.tv_usec = intr;
  Value.it_value.tv_sec = 0;
  Value.it_value.tv_usec = intr;
  setitimer(ITIMER_REAL, &Value, &OValue);
  signal(SIGALRM, NullFunction);

} /* end of SyncInit */


void GameInit(int mode)
{
  int  p;

  SyncSet(interval = IntervalTime);
  timer = 0;
  PixInit();

  mis_count = 0;
  mis_left = mis_right = True;
  damage = 0;
  if (mode) {
    level = 1;
    score = 0;
  }

  for (p = 0; p < M_MAX*2; p++) {
    missile[p][0].flag = NO_USE;
    missile[p][1].flag = NO_USE;
    expl[p].flag       = NO_USE;
  }

} /* end of GameInit */


void M_Set(int m, float v, float sx, float sy, float dx, float dy)
{
  float f;
  int   p;

  for (p = M_MAX*m; p < M_MAX*(m+1); p++) {
    if (missile[p][0].flag == NO_USE) {
      break;
    }
  }
  if (p < M_MAX*(m+1)) {
    missile[p][0].x = missile[p][1].x = sx;
    missile[p][0].y = missile[p][1].y = sy;
    f = sqrt((dx-sx)*(dx-sx)+(dy-sy)*(dy-sy));
    if ((f == 0)||(v == 0)) {
      M_Exp(dx, dy);
    } else {
      missile[p][0].vx = missile[p][1].vx = v*(dx-sx)/f;
      missile[p][0].vy = missile[p][1].vy = v*(dy-sy)/f;
      missile[p][0].dx = missile[p][1].dx = dx;
      missile[p][0].dy = missile[p][1].dy = dy;
      missile[p][0].flag = missile[p][1].flag = MISSILE;
      missile[p][0].state = 0;
      if (v == VEROCITY1) {
    missile[p][1].state = 40;
      } else if (v == VEROCITY2) {
    missile[p][1].state = 20;
      } else if (v == VEROCITY3) {
    missile[p][1].state = 13;
      } else if (v == VEROCITY4) {
    missile[p][1].state = 10;
      } else if (v == VEROCITY5) {
    missile[p][1].state = 5;
      }
    }
  }

} /* M_Set */


void M_Exp(float sx, float sy)
{
  int  p;

  for (p = 0; p < M_MAX*2; p++) {
    if (expl[p].flag == NO_USE) {
      break;
    }
  }
  if (p < M_MAX*2) {
    expl[p].x = expl[p].dx = sx;
    expl[p].y = expl[p].dy = sy;
    expl[p].vx = expl[p].vy = 0;
    expl[p].flag = EXPLOSION;
    expl[p].state = RANGE;
  }

} /* end of M_Exp */


void M_Draw(int mode)
{
  double d1, d2, d3, d4;
  int    p, q;

  for (p = 0; p < M_MAX*2; p++) {
    if (missile[p][0].flag == MISSILE) {
      for (q = 0; q < 2; q++) {
    if (missile[p][q].state > 0) {
      missile[p][q].state--;
    } else if (missile[p][q].state == 0) {
      missile[p][q].x += missile[p][q].vx;
      missile[p][q].y += missile[p][q].vy;
      d1 = (missile[p][q].dx-missile[p][q].x);
      d2 = (missile[p][q].dy-missile[p][q].y);
      d3 = missile[p][q].vx/2*missile[p][q].vx/2;
      d4 = missile[p][q].vy/2*missile[p][q].vy/2;
      if (d1*d1+d2*d2 <= d3+d4+1) {
        missile[p][q].x = missile[p][q].dx;
        missile[p][q].y = missile[p][q].dy;
        missile[p][q].state = -1;
        if (q == 0) {
          M_Exp(missile[p][0].dx, missile[p][0].dy);
        }
      }
    }
      }
      if (missile[p][1].state < 0) {
    missile[p][0].flag = missile[p][1].flag = NO_USE;
      }
    }
  }
  for (p = 0; p < M_MAX*2; p++) {
    if (expl[p].flag == EXPLOSION) {
      if (expl[p].state >= 0) {
    expl[p].state--;
    if (expl[p].state >= RANGE/2) {
      if (mode) {
        if (expl[p].y+(RANGE-expl[p].state) >= POINT1_Y) {
          damage += 1;
          XDrawPoint(dpy, pix1, gc_w,
             expl[p].x - 5 + rand()%10, POINT1_Y - 1 + rand()%7);
          XDrawPoint(dpy, pix1, gc_b,
             expl[p].x - 5 + rand()%10, POINT1_Y - 1 + rand()%7);
        }
        if (mis_left) {
          d1 = expl[p].x-POINT1_X;
          d2 = expl[p].y-POINT1_Y;
          if (d1*d1+d2*d2 <= (RANGE-expl[p].state)*(RANGE-expl[p].state)) {
        mis_left = False;
        M_Exp(POINT1_X, POINT1_Y);
        XFillArc(dpy, pix1, gc_b,
             POINT1_X-10, POINT1_Y-10, 20, 20,
             0*64, 360*64);
          }
        }
        if (mis_right) {
          d1 = expl[p].x-POINT2_X;
          d2 = expl[p].y-POINT2_Y;
          if (d1*d1+d2*d2 <= (RANGE-expl[p].state)*(RANGE-expl[p].state)) {
        mis_right = False;
        M_Exp(POINT2_X, POINT2_Y);
        XFillArc(dpy, pix1, gc_b,
             POINT2_X-10, POINT2_Y-10, 20, 20,
             0*64, 360*64);
          }
        }
      }
      for (q = M_MAX; q < M_MAX*2; q++) {
        if (missile[q][0].flag == MISSILE) {
          d1 = missile[q][0].x-expl[p].x;
          d2 = missile[q][0].y-expl[p].y;
          if (d1*d1+d2*d2 <= (RANGE-expl[p].state)*(RANGE-expl[p].state)) {
        M_Exp(missile[q][0].x, missile[q][0].y);
        missile[q][0].flag = missile[q][1].flag = NO_USE;
        if (mode&&(missile[q][0].y < POINT1_Y - 1)) {
          score += 100;
        }
          }
        }
      }
    }
      } else {
    expl[p].flag = NO_USE;
      }
    }
  }

} /* end of M_Draw */


void M_Disp(int mode)
{
  char buf[64];
  int  p, r;

  XCopyArea(dpy, pix1, pix0, gc_b, 0, 0, W0_WIDTH, W0_HEIGHT, 0, 0);
  for (p = 0; p < M_MAX*2; p++) {
    if (missile[p][0].flag == MISSILE) {
      if (timer % 10 == 0) {
    XDrawArc(dpy, pix0, gc_w,
         missile[p][0].x-2, missile[p][0].y-2, 4, 4, 0*64, 360*64);
      }
      XDrawLine(dpy, pix0, gc_w,
        missile[p][0].x, missile[p][0].y,
        missile[p][1].x, missile[p][1].y);
    }
  }
  for (p = 0; p < M_MAX*2; p++) {
    if (expl[p].flag == EXPLOSION) {
      if ((expl[p].state > RANGE/2)||(expl[p].state % 2 == 0)) {
    r = expl[p].state > RANGE/2 ? RANGE-expl[p].state : expl[p].state;
    XFillArc(dpy, pix0, gc_w,
         expl[p].x-r, expl[p].y-r, r*2, r*2, 0*64, 360*64);
      }
    }
  }
  if (mode) {
    if ((0 <= timer % 50)&&(timer % 50 < 30)&&(timer % 2 == 0)&&
    (75*4 <= damage)&&(damage < 100*4)) {
      XDrawString(dpy, pix0, gc_w, 94, 50, "D  A  N  G  E  R", 16);
      XDrawString(dpy, pix0, gc_w, 96, 51, "D  A  N  G  E  R", 16);
    }
    sprintf(buf, "DAMAGE : %3d %%", damage/4);
    XDrawString(dpy, pix0, gc_b, 10, W0_HEIGHT-34, buf, strlen(buf));
    sprintf(buf, "level  : %3d", level);
    XDrawString(dpy, pix0, gc_b, 10, W0_HEIGHT-18, buf, strlen(buf));
    sprintf(buf, "score  : %6d pts.", score);
    XDrawString(dpy, pix0, gc_b, 10, W0_HEIGHT-6, buf, strlen(buf));
  } else {
    sprintf(buf, "your score is %6d pts.(level : %3d)", score, level);
    XDrawString(dpy, pix0, gc_b, 68, W0_HEIGHT-28, buf, strlen(buf));
    sprintf(buf, "high score is %6d pts.(%s)", hi_score, hi_user_name);
    XDrawString(dpy, pix0, gc_b, 68, W0_HEIGHT-10, buf, strlen(buf));
  }

} /* end of M_Disp */


void    Score_Load()
{
  FILE *fp;
  int  i;

  fp = fopen(SCORE_FILE, "rb");
  if (fp == NULL) {
    hi_score = 1000;
    sprintf(hi_user_name, "none");
    return;
  }
  hi_score = fgetc(fp);
  hi_score = hi_score*0x0100 + fgetc(fp);
  hi_score = hi_score*0x0100 + fgetc(fp);
  hi_score *= 100;
  for (i = 0; i < 16; i++) {
    hi_user_name[i] = fgetc(fp);
  }
  fclose(fp);

} /* end of Score_Load */


void    Score_Save()
{
  FILE *fp;
  int  i;

  fp = fopen(SCORE_FILE, "wb");
  if (fp == NULL) {
    fprintf(stderr, "Can't open hi_score file.\n");
    return;
  }
  fputc(((hi_score/100) >> 16) & 0xff, fp);
  fputc(((hi_score/100) >>  8) & 0xff, fp);
  fputc(((hi_score/100) >>  0) & 0xff, fp);
  for (i = 0; i < 16; i++) {
    fputc(hi_user_name[i], fp);
  }
  fputc('\n', fp);
  fclose(fp);

} /* end of Score_Save */