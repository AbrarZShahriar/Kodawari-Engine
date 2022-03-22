#pragma once

#define MAX_ENTITIES

#include <glm/vec3.hpp>

struct movement
{
    glm::vec3 Position{};
    
    glm::vec3 Direction{};
    float Speed{};
};
struct game_controller
{
    int Up;
    int Down;
    int Left;
    int Right;
};

struct entity_megastruct
{
    /*
	* EntityID
	* IsActive
	* IsEnemy
	* IsBullet
	* IsTurret
	* IsFLoor
	* Position
	* Velocity
	* HP
	* 
	* 
	* 
	* 
	*/
    movement Movement{};
    glm::vec3 Scale = {};
    game_controller *Controller;
    int EntityID{};
    

	bool IsActive{};
    
    bool IsEnemy{};
    bool IsPlayer{};
    bool IsBullet{};
    
    bool IsTurret{};
    bool IsFloor{};
    
    
    int HP{};
		

};




void
controller_reset(game_controller *G);
void
move_entity_0(entity_megastruct *Entity);

struct game_state
{
    /*
	*
	* StartMenu
	* Running
	* 
	* EndScreen
	* 
	* 
	* 
	* 
	*/
    bool StartMenu = true;
    bool Running = false;
    int Score = 0;
};