(ns ftlm.disco.middlewares.picture-jitter
  (:require
   [quil.core :as q :include-macros true]))

(defn default-position
  "Default position configuration: zoom is neutral and central point is
  `width/2, height/2`."
  []
  {:position [(/ (q/width) 2.0) (/ (q/height) 2.0)]
   :zoom 1})

(defn draw
  [user-draw state]
  (q/push-matrix)
  (let [picture-jitter (:picture-jitter state)
        zoom (:zoom picture-jitter)
        pos (:position picture-jitter)]
    (q/scale zoom)
    (q/with-translation
      [(- (/ (q/width) 2 zoom) (first pos))
       (- (/ (q/height) 2 zoom) (second pos))]
      (user-draw state)))
  (q/pop-matrix))

(defn add-vec
  "Add two or more vectors together"
  [& args]
  (when (seq args) (apply mapv + args)))

(defn mult-vec [& args]
  (when (seq args)
    (apply mapv * args)))

(defn update-jitter-state
  [jitter]
  (let [v (q/random-2d)
        v (mult-vec v [2 2])
        intensity (+ 1 (:intensity jitter 1))
        zoom-intensity (:zoom-intensity jitter 0)
        v (mult-vec v [intensity intensity])]
    (-> jitter
        (update :position add-vec v)
        (update :zoom + (* zoom-intensity (q/random -0.1 0.1))))))

(defn update-state
  [user-update state]
  (let [picture-jitter (:picture-jitter state)]
    (cond-> (user-update state)
      (:jitter? picture-jitter)
      (update :picture-jitter update-jitter-state))))

(defn setup-picture-jitter
  [user-setup user-settings]
  (let [initial-state (merge (default-position)
                             user-settings)]
    (update-in
     (user-setup)
     [:picture-jitter]
     #(merge initial-state %))))

(defn picture-jitter
  [options]
  (let [user-settings (:picture-jitter options)
        user-draw (:draw options (fn [state]))
        user-update (:update options (fn [state]))
        user-setup (:setup options (fn [] {}))]
    (def user-setup user-setup)
    options
    (assoc options
           :setup (partial setup-picture-jitter user-setup user-settings)
           :update (partial update-state user-update)
           :draw (partial draw user-draw))))
