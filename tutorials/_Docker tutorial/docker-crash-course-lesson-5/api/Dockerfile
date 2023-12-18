FROM node:17-alpine

# installing nodemon globally for "volumes" while adding it to the package.json at the same time
RUN npm install -g nodemon

WORKDIR /app

# copying to installing before importing all the source code
COPY package.json .

# a node command for installing project dependencies
RUN npm install 

COPY . .

# required for docker desktop port mapping
EXPOSE 4000

# instead of node app.js
# CMD ["node", "app.js"]

# Running the dev inside the package.json
CMD ["npm", "run", "dev"]