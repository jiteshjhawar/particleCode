/*
 * Screen.h
 *
 *  Created on: 05-05-2015
 *      Author: jitesh
 */

#ifndef SCREEN_H_
#define SCREEN_H_

#include <SDL2/SDL.h>
#include <cstring>
#include <iomanip>


class Screen {
public:
	//int m_l;
	const static int SCREEN_WIDTH = 500;
	const static int SCREEN_HEIGHT = 500;
private:
	SDL_Window *m_window;
	SDL_Renderer *m_renderer;
	SDL_Texture *m_texture;
	Uint32 *m_buffer;
public:
	Screen();
	//Screen(int m_l);
	void update();
	void setPixel(int x, int y, Uint8 red, Uint8 green, Uint8 blue);
	bool init();
	bool processEvents();
	void clear();
	void close();
};


#endif /* SCREEN_H_ */
