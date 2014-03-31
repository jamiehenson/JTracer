/*  
	Ray Tracer
	Jamie Henson

	A recursive raytracer that uses cosine-weighted Monte Carlo 
	hemispherical sampling.

	Contains:
	- Framework: Standard camera, multi-threading, external scene file loader
	- Primitives: Plane, sphere, triangle
	- Materials: Lambertian, specular, refractive
	- Refractive materials: Glass, air, water, diamond
	- Other: Ambient occlusion, global illumination, colour bleeding
	
	Compile using:
	g++ jtracer.cpp -O3 -std=c++11 -Wall -pedantic -fopenmp -o jtracer

	Run using:
	./jtracer SAMPLESIZE RESOLUTION SCENEFILE OUTPUTFILE*
	(* denotes an optional parameter)

	For example:
	./jtracer 32 400 casino casinopic
	Will export a 400x400 image at 32 samples per pixel from casino.scene, to casinopic.ppm

	If you give no parameters, the program will use "default.scene" and output "default.ppm",
	with a resolution of 512x512 at 16 samples per pixel.

	Scene files are of type ".scene"

	Based upon several small ray-tracer implementations
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
using namespace std;

// Our computable version of infinity (since we can't actually use infinity)
const double massive = 1e9;

// A tiny value, slightly more than 0, used for intersection testing
const double tiny = 1e-10;

/* 
	Fundamental Vector structure, containing base definition (x,y,z),
	and several operators that allow for the arithmetic combination of
	two vectors.
*/
struct Vector 
{
	// Vector coordinates, also doubles up as the colour (r,g,b)
	double x, y, z;

	// Constructor, set all to 0
	Vector(double x0 = 0, double y0 = 0, double z0 = 0)
	{ 
		x = x0; y = y0; z = z0; 
	}
	
	// Vector addition
	Vector operator + (const Vector &other) const 
	{ 
		return Vector(x+other.x, y+other.y, z+other.z); 
	}

	// Vector subtraction
	Vector operator - (const Vector &other) const 
	{ 
		return Vector(x-other.x, y-other.y, z-other.z); 
	}

	// Scalar vector multiplication
	Vector operator * (double other) const 
	{ 
		return Vector(x*other, y*other, z*other); 
	}

	// Scalar vector division
	Vector operator / (double other) const 
	{ 
		return Vector(x/other, y/other, z/other); 
	}

	// Cross product
	Vector operator % (Vector &other)
	{
		return Vector(y*other.z - z*other.y, z*other.x - x*other.z, x*other.y - y*other.x);
	}

	// Dot product
	double dot(const Vector &other) const 
	{ 
		return x*other.x + y*other.y + z*other.z; 
	}

	// Vector multiplication
	Vector vecmult(const Vector &other) const 
	{ 
		return Vector(x*other.x, y*other.y, z*other.z); 
	}

	// Normalised version of a vector
	Vector& normalise()
	{ 
		return *this = *this * (1/sqrt(x*x + y*y + z*z)); 
	}

	// Magnitude
	double mag() 
	{ 
		return sqrt(x*x + y*y + z*z); 
	}
};

/*
	Structure for a Ray, containing its origin vector and directional vector.
	Ray vectors are initialised as 0 by default.
*/
struct Ray 
{
	Vector origin, direction;
	Ray(Vector o = 0, Vector d = 0) 
	{ 
		origin = o, direction = d.normalise(); 
	}
};

// Determine the refractive index of a refractive surface
inline double SurfaceType(string name)
{
	if (name == "glass") return 1.52;
	else if (name == "air") return 1.0003;
	else if (name == "water") return 1.33;
	else if (name == "diamond") return 2.4;
	else return 1; // Vacuum
}

/* 
	RGB pixel colours can only be expressed in the range 0-255.
	Therefore, all values must be clamped to this range
*/
inline int clamper(double val)
{ 
  if (val < 0) return (int) 0;
  else if (val > 255) return (int) 255;
  else return (int) val;
}

// Governing "shape" constructor
class Shape 
{
	public:
	Vector colour;
	double emission;
	int type;
	string material;
	void assignMaterial (Vector col = 0, double emi = 0, int typ = 0, string mat = "")
	{ 
		colour = col; 
		emission = emi; 
		type = typ; 
		material = mat;
	}
	virtual double intersection(const Ray&) const = 0;
	virtual Vector normal(const Vector&) const = 0;
};

// The "plane" primitive, of type "Shape"
class Plane : public Shape 
{
public:
	// Consists of a normal vector and a position along that vector
	Vector norm;
	double pos;

	// Initialise plane with zero values
	Plane(double pos0 = 0, Vector n0 = 0) 
	{ 
		pos = pos0; 
		norm = n0; 
	}

	// Ray-Plane intersection
	double intersection(const Ray& ray) const 
	{
		double d = norm.dot(ray.direction);
		if(d == 0) return 0;
		double t = -(norm.dot(ray.origin) + pos) / d;
		if (t > tiny) return t;
		else return 0;
	}

	// No need to calculate the normal, it is supplied in the plane definition,
	// but we still need this function to conform to the Shape constructor
	Vector normal(const Vector& p) const 
	{ 
		return norm; 
	}
};

// The "triangle" primitive, of type "Shape"
class Triangle : public Shape
{
public:
	// The three points of the triangle
	Vector a,b,c;

	// All points initialised to be (0,0,0)
	Triangle(Vector a0 = 0, Vector b0 = 0, Vector c0 = 0)
	{
		a = a0;
		b = b0;
		c = c0;
	}

	//Ray-Triangle intersection using the Moller-Trumbore algorithm
	double intersection(const Ray& ray) const
	{
		// Using point "a" as origin, edge1 and edge2 come off it
		Vector edge1 = b - a;
		Vector edge2 = c - a;
		Vector rd = ray.direction;
		Vector pv = rd % edge2;
		Vector tv = ray.origin - a;
		Vector qv = tv % (edge1);
		
		// Test for determinant
		double det = edge1.dot(pv);
		
		// If determinant is great than 0, triangle is front-facing
		if (det > 0)
		{
			// Compute u, must be between 0 and 1 to be barycentric
			double u = tv.dot(pv) / det;
			if (u < 0 || u > 1) return 0;

			// Compute v, must be between 0 and 1 to be barycentric
			double v = rd.dot(edge1) / det;
			if (v < 0 || u + v > 1) return 0;

			// If both tests pass, we can now compute t
			double t = edge2.dot(qv) / det;
			if (t > tiny) return t;
			else return 0;
		}
		else // The triangle is back-facing.
		{
			// If triangle and ray are parallel
			if (det > -tiny && det < tiny) return 0;
		}
		return 0;
	}

	Vector normal(const Vector& p) const 
	{ 
		// The cross product of the two edges attached to A
		Vector bdiff = b - a;
		Vector cdiff = c - a;
		return bdiff % cdiff;
	}
};

// The "sphere" primitive, of type "Shape"
class Sphere : public Shape 
{
public:
	// Consists of a position vector, and a radius
	Vector pos;
	double rad;

	// Initialised to have a radius of 0 and a position vector of (0,0,0)
	Sphere(double rad0 = 0, Vector pos0 = 0) 
	{ 
		pos = pos0; 
		rad = rad0; 
	}

	// Ray-sphere intersection using the quadratic formula
	double intersection(const Ray& ray) const 
	{
		// Because of normalisation, a = 1.
		double a = 1;
		double b = ((ray.origin - pos) * 2).dot(ray.direction);
		double c = (ray.origin - pos).dot((ray.origin - pos)) - (rad*rad);
		double dis = b*b - 4*a*c;
		if (dis < 0) return 0;
		else dis = sqrt(dis);
		double x1 = -b + dis;
		double x2 = -b - dis;
		if (x2 > tiny) return x2 / (2*a);
		else if (x1 > tiny) return x1 / (2*a);
		else return 0;
	}

	Vector normal(const Vector& pos2) const 
	{
		return Vector((pos2.x-pos.x)/rad, (pos2.y-pos.y)/rad, (pos2.z-pos.z)/rad).normalise();
	}
};

/* 
	Function to set up the camera, and the ray for firing.
	Contains controls for manipulating the camera, and calculates 
	ray direction given the input position in the picture. Outputs the
	computed ray ready for firing into the scene.
*/
inline Ray setupray(const double x, const double y, int sizei) 
{
	// Picture size
	double size = (double) sizei;

	// Camera manipulation controls
	double xrotation = 0; // X rotation, +ve = right
	double yrotation = 0; // Y rotation, +ve = down
	double xscale = 1; // X scaling, high = compression
	double yscale = 1; // Y scaling, high = compression
	double zoom = 0; // Zoom, high = closer (you can also adjust FOV)

	// Set up a ray
	Ray ray;
	ray.origin = (Vector(0,0,0));

	// Calculate its direction
	double fov = 0.8; // Field of view (has a zoom effect, minimise tweaking)
	double camx = ((2*yscale*x - size) / size) * tan(fov) - yrotation;
	double camy = ((2*xscale*y - size) / size) * tan(fov) + xrotation;
	Vector cam = Vector(camx,camy,-1 - zoom);
	ray.direction = (cam - ray.origin).normalise();

	// Return the ray
	return ray;
}

// Cosine-weighted hemispherical Monte Carlo sampler
// (from Ryan Driscoll's article on "Better Sampling")
Vector hemisampler(double random) 
{
	double pi = 3.14159265359;
	const float rnd = sqrt(random);
    const float theta = 2 * pi * random; // Disc point modelling
    const float x = rnd * cos(theta);
    const float y = rnd * sin(theta);
    return Vector(x, y, sqrt(max(0.0, 1 - random))); // Project to hemisphere
}

/*
	The primary ray-tracing function that runs recursively. The primary steps include:
		1. Reference a single ray
		2. Test for intersections between that ray and all the components in the scene
		3. Choose the closest component
		4. Determine the surface type of that component (diffuse, specular, transparent) and compute the colour
			at that point
		5. Recursively loop, tracing the rays that are emitted as reflections from the initial ray
		6. Stop when the depth threshold is reached
		7. Depending on the surface type, return the final colour.
*/

void raytrace(Ray &ray, const vector<Shape*>& scene, int depth, Vector& colour) 
{
	// Limit of recursive reflection tracing, 6 chosen through compromise of speed vs. observable changes
	if (depth > 6) return;

	// Set up the variables for scene testing
	double t; // The intersection distance
	int id = -1; // A value we know is false
	double dist = massive; // The value to decrement with known values of t
	const int size = scene.size();

	// Testing for intersections, with all elements in the scene
	for(int x = 0; x < size; x++) 
	{
		t = scene[x]->intersection(ray); // Compute the intersection
		if (t > tiny && t < dist) // Iteratively choose the closest
		{ 
			dist = t; 
			id = x; // Reference the component by its ID for later use
		}
	}
	if (id == -1) return;

	// Acquire ray point and shape normal at that point
	Vector rp = ray.origin + ray.direction * dist; // P = O + tD
	Vector N = scene[id]->normal(rp);

	// Set the ray origin to be the ray point
	ray.origin = rp;

	// Add emission values, currently adds the same value to R, G and B for a "white" light
	colour = colour + Vector(scene[id]->emission,scene[id]->emission,scene[id]->emission);

	// In the case of a Lambertian surface
	if(scene[id]->type == 0) 
	{
		// Colour balancing (more is more)
		double rBalance = 1;
		double gBalance = 1;
		double bBalance = 1;

		// Add hemispherical sampler vector to normal
		ray.direction = (N + hemisampler((double) rand() / RAND_MAX));
		double costheta = ray.direction.dot(N);
		Vector next = Vector();
		raytrace(ray,scene,depth+1,next);

		// Aggregate of colours
		colour.x += costheta * (next.x * scene[id]->colour.x) * rBalance/10;
		colour.y += costheta * (next.y * scene[id]->colour.y) * gBalance/10;
		colour.z += costheta * (next.z * scene[id]->colour.z) * bBalance/10;
	}

	// In the case of a specular surface
	if(scene[id]->type == 1) 
	{
		// Dot product of in-ray and normal
		double cosI = ray.direction.dot(N);

		// Output ray
		ray.direction = (ray.direction-N*(cosI*2)).normalise();

		// Define new colour vector, recursively loop back round
		Vector next = Vector();
		raytrace(ray, scene, depth+1, next);
		colour = colour + next;
	}

	// In the case of a refractive surface
	if(scene[id]->type == 2) 
	{
		// Get the refractive index of the supplied material
		double refIndex = SurfaceType(scene[id]->material);

		// How shiny do you want your transparent shapes to be?
		double shiny = 1.2;
		
		// Is the ray going in from the outside?
		double cosI = N.dot(ray.direction);
		if(cosI > 0) 
		{
			// If so, flip normal and refractive index
			N = N * -1;
			refIndex = 1/refIndex;
		}

		// Recalculate dot product of N and direction
		cosI = -(N.dot(ray.direction));

		// Test for total internal reflection
		double cosT = sqrt(1.0 - (1/(refIndex*refIndex)) * (1.0-(cosI*cosI)));
		
		// Does it refract?
		if (cosT > 0) 
		{
			// Yep. It does. Find direction using Snell's Law and reflection calculation:
			ray.direction = ((ray.direction * 1/refIndex) + (N * (1/refIndex * cosI - sqrt(cosT)))).normalise();
			Vector next = Vector();
			raytrace(ray, scene, depth+1, next);
			colour = colour + next * shiny;
		}

		// Otherwise, it reflects. Use the same code as for a specular surface here.
		cosI = 2 * ray.direction.dot(N);
		ray.direction = (ray.direction-N * (cosI)).normalise();
		Vector next = Vector();
		raytrace(ray, scene, depth+1, next);
		colour = colour + next;
		return;
	}
}

// Scene file parser (takes type *.scene)
vector<Shape*> createScene(string scenename)
{
	// The container for all shapes
	vector<Shape*> scene;

	// Specify construction for adding to the vector "scene", the type "Shape" can accept sub-types
	// Plane, Triangle and Sphere, along with general colour, emission and material details
	auto add=[&scene](Shape* shape, Vector colour, double emission, int type, string material) 
	{
		shape->assignMaterial(colour,emission,type,material);
		scene.push_back(shape);
	};

	// Open up the scene file
	ifstream scenefile(scenename);

	// While it's still able to be traversed...
	if(scenefile) 
	{
	  string seg="";
	  // Parse scene file, line by line
	  while(getline(scenefile, seg)) 
	  {
	  	vector<string> features;
	  	stringstream line(seg);
	  	if (seg == "" || seg.find("\\") == 0) continue; // Don't parse empty lines or comments!
	  	while (getline(line,seg,',')) features.push_back(seg); // Split line by comma delimiter

	  	// Type: Sphere
	  	if (features[0] == "Sphere")
	  	{
	  		double rad = atof((features[1].substr(4,features[1].length()-5)).c_str());
	  		double posx = atof((features[2].substr(5,features[2].length()-6)).c_str());
	  		double posy = atof((features[3].substr(5,features[3].length()-6)).c_str());
	  		double posz = atof((features[4].substr(5,features[4].length()-6)).c_str());
	  		double red = atof((features[5].substr(4,features[5].length()-5)).c_str());
	  		double green = atof((features[6].substr(6,features[6].length()-7)).c_str());
	  		double blue = atof((features[7].substr(5,features[7].length()-6)).c_str());
	  		double emi = atof((features[8].substr(4,features[8].length()-5)).c_str());
	  		int sur = atoi((features[9].substr(4,features[9].length()-5)).c_str());
	  		string mat = (features[10].substr(4,features[10].length()-5));

	  		// Add parsed values to new instance of Sphere object
	  		add(new Sphere(rad,Vector(posy,posx,posz)),Vector(red,green,blue),emi,sur,mat);

	  		features[0] = "";
	  	}

	  	// Type: Plane
	  	else if (features[0] == "Plane")
	  	{
	  		double pos = atof((features[1].substr(4,features[1].length()-5)).c_str());
	  		double normx = atof((features[2].substr(6,features[2].length()-7)).c_str());
	  		double normy = atof((features[3].substr(6,features[3].length()-7)).c_str());
	  		double normz = atof((features[4].substr(6,features[4].length()-7)).c_str());
	  		double red = atof((features[5].substr(4,features[5].length()-5)).c_str());
	  		double green = atof((features[6].substr(6,features[6].length()-7)).c_str());
	  		double blue = atof((features[7].substr(5,features[7].length()-6)).c_str());
	  		double emi = atof((features[8].substr(4,features[8].length()-5)).c_str());
	  		int sur = atoi((features[9].substr(4,features[9].length()-5)).c_str());
	  		string mat = (features[10].substr(4,features[10].length()-5));

			// Add parsed values to new instance of Plane object
	  		add(new Plane(pos,Vector(normx,normy,normz)),Vector(red,green,blue),emi,sur,mat);

	  		features[0] = "";
	  	}

	  	//Type: Triangle
	  	else if (features[0] == "Triangle")
	  	{
	  		double ax = atof((features[1].substr(3,features[1].length()-4)).c_str());
	  		double ay = atof((features[2].substr(3,features[2].length()-4)).c_str());
	  		double az = atof((features[3].substr(3,features[3].length()-4)).c_str());
	  		double bx = atof((features[4].substr(3,features[4].length()-4)).c_str());
	  		double by = atof((features[5].substr(3,features[5].length()-4)).c_str());
	  		double bz = atof((features[6].substr(3,features[6].length()-4)).c_str());
	  		double cx = atof((features[7].substr(3,features[7].length()-4)).c_str());
	  		double cy = atof((features[8].substr(3,features[8].length()-4)).c_str());
	  		double cz = atof((features[9].substr(3,features[9].length()-4)).c_str());
	  		double red = atof((features[10].substr(4,features[10].length()-5)).c_str());
	  		double green = atof((features[11].substr(6,features[11].length()-7)).c_str());
	  		double blue = atof((features[12].substr(5,features[12].length()-6)).c_str());
	  		double emi = atof((features[13].substr(4,features[13].length()-5)).c_str());
	  		int sur = atoi((features[14].substr(4,features[14].length()-5)).c_str());
	  		string mat = (features[15].substr(4,features[15].length()-5));

	  		// Add parsed values to new instance of Triangle object
	  		add(new Triangle(Vector(ay,ax,az),Vector(by,bx,bz),Vector(cy,cx,cz)),Vector(red,green,blue),emi,sur,mat);

	  		features[0] = "";
	  	}

	  	//Type: Not something that the ray-tracer can use
	  	else
	  	{
	  		fprintf(stderr,"Unknown shape type detected: %s\n",features[0].c_str());
	  	}
	  }
	}

	// Return collection of parsed scene objects
	return scene;
}

int main(int argc, char *argv[]) 
{
	int width, height; // Resolution
	double samples; // Times to trace per pixel
	string imagename, scenename, name;

	if (argc == 5) // All details specified
	{
		samples = atof(argv[1]);
		width = atoi(argv[2]);
		scenename = argv[3];
		scenename = scenename + ".scene";
		imagename = argv[4];
		imagename = imagename + ".ppm";
	}
	else if (argc == 4) // No output name specified
	{
		samples = atof(argv[1]);
		width = atoi(argv[2]);
		name = argv[3];
		scenename = name + ".scene";
		imagename = name + ".ppm";
	}
	else
	{
		width = 512;
		imagename = "default.ppm";
		scenename = "default.scene";
		samples = 16;
	}

	height = width;

	srand(time(NULL));

	// Create the scene to render, loaded from an external file of type ".scene"
	vector<Shape*> scene = createScene(scenename);
	
	// Define the image in terms of pixels
	Vector **pixels = new Vector*[width];
	for (int i=0; i < width; i++) 
	{
		pixels[i] = new Vector[height];
	}

	printf("-------------------------------------\nJTracer, by Jamie Henson (jh0422) for COMS30115.\n");
  	printf("Rendering... \nInput file: %s\nSamples per pixel: %d\nResolution: %dx%d\nOutput file: %s\n",scenename.c_str(),(int)samples,width,height,imagename.c_str());

  	// OpenMP parallel wizardry
	#pragma omp parallel for schedule(dynamic)

  	// Main raytracing loop (start with first row)
	for (int y = 0; y < height; y++) 
	{
		// Progress indicator
		fprintf(stderr,"\rProgress: %5.2f%%.",100.0*y/(width-1));
		// Increment through the columns
		for(int x = 0; x < width; x++) 
		{
			// Sampling "samples" times as we go, averaging result
			for(int s = 0; s < samples; s++) 
			{
				Vector colour; // Colour is automatically (0,0,0)

				// Set up a ray to trace for each sampling iteration				
				Ray ray = setupray(x,y,width);

				// Recursively trace at that location
				raytrace(ray,scene,0,colour);

				// Averaging the result and saving it in an overall 2D array of pixels
				pixels[x][y].x += colour.x * (1/samples);
				pixels[x][y].y += colour.y * (1/samples);
				pixels[x][y].z += colour.z * (1/samples);
			}
		}
	}

	// Output file management
	FILE *image = fopen(imagename.c_str(), "w");         
  	fprintf(image, "P3\n%d %d\n%d\n", width, height, 255);
	
	// Loop through the 2D pixel array and print accordingly to the PPM file
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++) 
		{
			// Clamp pixels between 0 and 255, in cases where the value lies outside
			fprintf(image,"%d %d %d ", clamper(pixels[i][j].x), clamper(pixels[i][j].y), clamper(pixels[i][j].z));
		}
	}

	// Finished!.
	fclose(image);
	printf("\nJob done!\n");
	return 0;
}