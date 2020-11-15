const express = require("express");
const fetch = require("node-fetch");
const ffmpeg = require("ffmpeg");
const fs = require("fs");
const fs_extra = require("fs-extra");
const tf = require("@tensorflow/tfjs-node");
const posenet = require("@tensorflow-models/posenet");
const { createCanvas, Image } = require("canvas");
const fileUpload = require("express-fileupload");
const cors = require("cors");
const bodyParser = require("body-parser");
const morgan = require("morgan");
const _ = require("lodash");

const imageScaleFactor = 0.5;
const outputStride = 16;
const flipHorizontal = false;
let net;

const calculatePose = async (path, idx) => {
  const img = new Image();
  img.src = path;
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const input = tf.browser.fromPixels(canvas);
  const pose = await net.estimateSinglePose(
    input,
    imageScaleFactor,
    flipHorizontal,
    outputStride
  );
  pose.index = idx;
  return pose;
};

const sendSuccess = async (results, exercise, token) => {
  fetch(`http://localhost:8000/exercises/${exercise.id}/video_results/`, {
    method: "POST",
    body: JSON.stringify({
      error: false,
      results,
    }),
    headers: {
      "Content-Type": "application/json",
      Authorization: `Token ${token}`,
    },
  });
};
const sendError = async (results, exercise, token) => {
  fetch(`http://localhost:8000/exercises/${exercise.id}/video_results/`, {
    method: "POST",
    body: JSON.stringify({
      error: true,
      results,
    }),
    headers: {
      Authorization: `Token ${token}`,
    },
  });
};

const processVideo = async (filePath, exercise, points, token) => {
  const process = new ffmpeg(filePath);
  process.then(
    function (video) {
      video.fnExtractFrameToJPG(
        "./assets/frames/",
        {
          every_n_frames: 2,
          file_name: "",
        },
        async () => {
          console.log("frames getted :D");
          try {
            const frames = await fs.promises.readdir("./assets/frames/");
            const poses = [];
            for (const framename of frames) {
              const pose = await calculatePose(
                `./assets/frames/${framename}`,
                parseInt(framename.slice(1, -4))
              );
              poses.push(pose);
            }
            poses.sort((a, b) => a.index - b.index);
            poses_tracked_points = {
              poses: poses.map((pose) => {
                const pose_tracked_points = [];
                for (let i = 0; i < points.length; i++) {
                  const point = points[i];
                  const center = pose.keypoints.find(
                    (pose_point) => pose_point.part === point.center
                  );
                  const left_point = pose.keypoints.find(
                    (pose_point) => pose_point.part === point.left_point
                  );
                  const right_point = pose.keypoints.find(
                    (pose_point) => pose_point.part === point.right_point
                  );
                  const angle = find_angle(
                    left_point.position,
                    right_point.position,
                    center.position
                  );
                  if (point.min_angle) {
                    point.min_angle =
                      point.min_angle < angle ? point.min_angle : angle;
                  } else {
                    point.min_angle = angle;
                  }

                  if (point.max_angle) {
                    point.max_angle =
                      point.max_angle > angle ? point.max_angle : angle;
                  } else {
                    point.max_angle = angle;
                  }

                  pose_tracked_points.push({
                    center,
                    left_point,
                    right_point,
                    angle,
                  });
                }
                return {
                  index: pose.index,
                  tracked_points: pose_tracked_points,
                };
              }),
              points,
            };
            sendSuccess(poses_tracked_points, exercise, token)
              .then((r) => console.log(r))
              .catch((r) => console.log(r));
          } catch (error) {
            sendError(`Error ${error}`, exercise, token)
              .then((r) => console.log(r))
              .catch((r) => console.log(r));
          }
        }
      );
    },
    function (err) {
      sendError("Error: " + err, exercise)
        .then((r) => console.log(r))
        .catch((r) => console.log(r));
    }
  );
};

/*
 * Calculates the angle ABC (in radians)
 *
 * A first point, ex: {x: 0, y: 0}
 * C second point
 * B center point
 */
function find_angle(A, B, C) {
  var AB = Math.sqrt(Math.pow(B.x - A.x, 2) + Math.pow(B.y - A.y, 2));
  var BC = Math.sqrt(Math.pow(B.x - C.x, 2) + Math.pow(B.y - C.y, 2));
  var AC = Math.sqrt(Math.pow(C.x - A.x, 2) + Math.pow(C.y - A.y, 2));
  return (
    (Math.acos((BC * BC + AB * AB - AC * AC) / (2 * BC * AB)) * 180) / Math.PI
  );
}

const app = express();

app.use(
  fileUpload({
    createParentPath: true,
  })
);

//add other middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(morgan("dev"));

app.get("/", async (req, res) => {
  try {
    const process = new ffmpeg("./assets/video/test.mp4");
    process.then(
      function (video) {
        video.fnExtractFrameToJPG(
          "./assets/frames/",
          {
            every_n_frames: 2,
            file_name: "",
          },
          async () => {
            console.log("frames getted :D");
            try {
              const frames = await fs.promises.readdir("./assets/frames/");
              const poses = [];
              for (const framename of frames) {
                const pose = await calculatePose(
                  `./assets/frames/${framename}`,
                  parseInt(framename.slice(1, -4))
                );
                poses.push(pose);
              }
              poses.sort((a, b) =>
                a.index > b.index ? 1 : b.index > a.index ? -1 : 0
              );
              res.json(poses);
            } catch (error) {
              console.error(error);
            }
          }
        );
      },
      function (err) {
        console.log("Error: " + err);
      }
    );
  } catch (e) {
    console.log("error :c");
    console.log(e.code);
    console.log(e.msg);
  }
});

app.post("/video", async (req, res) => {
  try {
    if (!req.files) {
      res.send({
        status: false,
        message: "No video uploaded",
      });
    } else {
      fs_extra.emptyDirSync("./assets/video/");
      fs_extra.emptyDirSync("./assets/frames/");
      const file = req.files.video;
      const filePath = "./assets/video" + file.name;
      file.mv(filePath);
      console.log("File moved");
      processVideo(
        filePath,
        JSON.parse(req.body.exercise),
        JSON.parse(req.body.points),
        req.body.token
      );
      console.log(req.body.token);
      res.status(200).end();
    }
  } catch (err) {
    res.status(500).send(err);
  }
});

app.listen(3000, async () => {
  try {
    net = await posenet.load({
      architecture: "MobileNetV1",
      outputStride: 16,
      inputResolution: 513,
      multiplier: 0.75,
    });
    console.log("Posenet loaded!", net);
  } catch (error) {
    console.error(error);
  }
  console.log("Escuchando en el 3000");
});

